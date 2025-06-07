"""
Heavily modified from REPA: https://github.com/sihyun-yu/REPA/blob/main/loss.py
"""

import pdb
import torch
import numpy as np
import torch.nn.functional as F


def mean_flat(x, temperature=1.0, **kwargs):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def sum_flat(x, temperature=1.):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def mse_flat(x, y, temperature=1, **kwargs):
    err = (x - y) ** 2
    return mean_flat(err)


def class_conditioned_sampling(labels):
    bsz = labels.shape[0]
    mask = ~(labels[None] == labels[:, None])
    # choices[mask] = -1 # remove self-sampling
    # Now randomly sample from the remaining choices
    weights = mask.float()
    weights_sum = weights.sum(dim=1, keepdim=True)
    if (weights_sum == 0).any():
        # In case there are no valid choices, fallback to uniform sampling
        choices = torch.randint(0, bsz, (bsz,), device=labels.device)
        # weights = torch.ones_like(choices).float()
    else:
        # Normalize weights to avoid division by zero
        weights = weights / weights_sum.clamp(min=1)
        # Sample from the available choices based on weights
        choices = torch.multinomial(weights, 1).squeeze(1)
    # Ensure no self-sampling
    assert (choices != torch.arange(bsz, device=labels.device)).all(), "Self-sampling detected in class_conditioned_sampling"
    return choices


class TripletSILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            contrastive_weight=1.0,
            null_class_idx=None,
            dont_contrast_on_unconditional=False,
            is_class_conditioned=False,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.null_class_idx = null_class_idx
        self.dont_contrast_on_unconditional = dont_contrast_on_unconditional
        self.is_class_conditioned = is_class_conditioned
        if self.dont_contrast_on_unconditional:
            assert self.null_class_idx is not None, "Null class index must be provided"
            
        self.temperature = contrastive_weight
        print(f"Using temperature of: {self.temperature}")

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def compute_triplet_loss_efficiently(self, x, y, labels=None):
        x = x.flatten(1)
        y = y.flatten(1)
        # Obtain positive samples and compute error
        y_pos = y
        pos_error = mean_flat((x - y_pos) ** 2)
        bsz = x.shape[0]
        choices = torch.tile(torch.arange(bsz), (bsz, 1)).to(x.device)
        choices.fill_diagonal_(-1.)
        choices = choices.sort(dim=1)[0][:, 1:]
        choices = choices[torch.arange(bsz), torch.randint(0, bsz-1, (bsz,))]
        y_neg = y[choices]
        # Compute error
        if self.dont_contrast_on_unconditional:
            non_nulls = labels != self.null_class_idx
        else:
            non_nulls = torch.ones_like(labels, dtype=torch.bool)
        neg_elem_error = ((x - y_neg) ** 2) * non_nulls.to(x.device).unsqueeze(-1)
        neg_elem_error = neg_elem_error
        neg_error = mean_flat(neg_elem_error) * bsz / non_nulls.sum() # rescale to account for null classes
        # Compute loss
        loss = pos_error - self.temperature * neg_error
        # return loss
        return {
            "loss": loss,
            "flow_loss": pos_error,
            "contrastive_loss": neg_error
        }
    
    def compute_class_conditioned_triplet_loss(self, x, y, labels=None):
        negative_idxs = class_conditioned_sampling(labels)
        bsz = x.shape[0]
        x = x.flatten(1)
        y = y.flatten(1)
        # Obtain positive samples and compute error
        y_pos = y
        pos_error = mean_flat((x - y_pos) ** 2)
        y_neg = y[negative_idxs]
        if self.dont_contrast_on_unconditional:
            non_nulls = labels != self.null_class_idx
        else:
            non_nulls = torch.ones_like(labels, dtype=torch.bool)
        neg_elem_error = ((x - y_neg) ** 2) * non_nulls.to(x.device).unsqueeze(-1)
        neg_error = mean_flat(neg_elem_error) * bsz / non_nulls.sum() # rescale to account for null classes
        # Compute loss
        loss = pos_error - self.temperature * neg_error
        return {
            "loss": loss,
            "flow_loss": pos_error,
            "contrastive_loss": neg_error
        }
    
    def contrastive_loss(self, pred, target_images, d_alpha_t, d_sigma_t, noises, labels=None, **kwargs):
        model_target = d_alpha_t * target_images + d_sigma_t * noises
        loss = self.compute_triplet_loss_efficiently(pred, model_target, labels)
        return loss
    
    def triplet_same_noise(self, pred, target_images, d_alpha_t, d_sigma_t, noises, labels=None):
        reconstructed_target = (pred - d_sigma_t * noises) / d_alpha_t
        loss = self.compute_triplet_loss(reconstructed_target, target_images, labels)
        return loss
    
    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        model_output, zs_tilde, labels = model(model_input, time_input.flatten(), **model_kwargs)
        # model_output, zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)
        # pdb.set_trace()
        denoising_loss = self.contrastive_loss(
            pred=model_output, 
            target_images=images, 
            d_alpha_t=d_alpha_t, 
            d_sigma_t=d_sigma_t, 
            noises=noises,
            labels=labels,
            noised_images=model_input,
            time=time_input, 
        )
        # projection loss
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)
        
        return denoising_loss, proj_loss
