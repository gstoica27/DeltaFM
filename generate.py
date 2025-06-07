# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import pdb
import torch
import torch.distributed as dist
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def create_npz_from_sample_folder_with_classes(sample_dir):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    # pdb.set_trace()
    samples_by_class = {}
    save_dir = os.path.join(sample_dir, "class_npzs")
    os.makedirs(save_dir, exist_ok=True)
    for fname in tqdm(os.listdir(sample_dir), desc="Building .npz file from samples"):
        if os.path.isdir(os.path.join(sample_dir, fname)):
            continue
        
        class_idx = fname.split("_")[0]
        sample_pil = Image.open(f"{sample_dir}/{fname}")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        if class_idx not in samples_by_class:
            samples_by_class[class_idx] = []
        samples_by_class[class_idx].append(sample_np)
        # samples.append(sample_np)
    # pdb.set_trace()
    for class_idx, samples in samples_by_class.items():
        samples = np.stack(samples)
        assert samples.shape == (len(samples), samples.shape[1], samples.shape[2], 3)
        npz_path = f"{save_dir}/{class_idx}.npz"
        np.savez(npz_path, arr_0=samples)
    print(f"Saved {len(os.listdir(save_dir))} npz files to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = True,
        z_dims = [int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    if args.interference_path is not None:
        # Load interference
        interference_vector = torch.load(args.interference_path, map_location=f'cuda:{device}').to(torch.float32)[None]
    else:
        interference_vector = None

    sample_dir = args.sample_dir
    # create interference method folder
    # pdb.set_trace()
    if args.interference_path is not None:
        interference_type = '-'.join(args.interference_path.split("/")[-1].split(".")[0].split('_')[1:])
        folder_name = f"{interference_type}"
        sample_dir = os.path.join(sample_dir, folder_name)
    
        model_name = ckpt_path.split("/")[-3]
        sample_dir = os.path.join(sample_dir, model_name)

    # create cfg folder
    if args.reduce_interference:
        float_to_str = lambda x: str(x).replace(".", "p")
        cfg_str = float_to_str(args.cfg_scale)
        guidance_high_str = float_to_str(args.guidance_high)
        nfe_str = str(args.num_steps)
        folder_name = "cfg-{}-guidance-{}-nfe-{}".format(cfg_str, guidance_high_str, nfe_str)
        # add intereference information
        if args.interference_weight is not None:
            folder_name += "-interference-weight-{}".format(float_to_str(args.interference_weight))
        if args.velocity_weight is not None:
            folder_name += "-velocity-weight-{}".format(float_to_str(args.velocity_weight))
        
        sample_dir = os.path.join(sample_dir, folder_name)
    # pdb.set_trace()
    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-guidance-H-{args.guidance_high}-L-{args.guidance_low}-heun"\
                    f"-{args.heun}-seed-{args.global_seed}-{args.mode}"
    # sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    sample_folder_dir = os.path.join(sample_dir, folder_name)
    sample_folder_classes_dir = os.path.join(sample_dir, folder_name + "_samples_per_class")
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(sample_folder_classes_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
        print(f"Saving .png samples by class at {sample_folder_classes_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    if rank == 0:
        print("Running Experiment with the following parameters:")
        print(f"  - model: {args.model}")
        print(f"  - ckpt: {args.ckpt}")
        print(f"  - sample_dir: {args.sample_dir}")
        print(f"  - num_classes: {args.num_classes}")
        print(f"  - per_proc_batch_size: {args.per_proc_batch_size}")
        print(f"  - num_fid_samples: {args.num_fid_samples}")
        print(f"  - mode: {args.mode}")
        print(f"  - cfg_scale: {args.cfg_scale}")
        print(f"  - guidance_low: {args.guidance_low}")
        print(f"  - guidance_high: {args.guidance_high}")
        print(f"  - path_type: {args.path_type}")
        print(f"  - num_steps: {args.num_steps}")
        print(f"  - heun: {args.heun}")
        print(f"  - interference_path: {args.interference_path}")
        print(f"  - interference_lambda: {args.interference_weight}")
        print(f"  - velocity_lambda: {args.velocity_weight}")
        print(f"  - reduce_interference: {args.reduce_interference}")
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            interference_vector=interference_vector,
            interference_lambda=args.interference_weight,
            velocity_lambda=args.velocity_weight,
            apply_reduction=args.reduce_interference,
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs)["samples"].to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs)["samples"].to(torch.float32)
            else:
                raise NotImplementedError()

            latents_scale = torch.tensor(
                [0.18215, 0.18215, 0.18215, 0.18215, ]
                ).view(1, 4, 1, 1).to(device)
            latents_bias = -torch.tensor(
                [0., 0., 0., 0.,]
                ).view(1, 4, 1, 1).to(device)
            samples = vae.decode((samples -  latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            # Save samples by class
            for i, (sample, sample_label) in enumerate(zip(samples, y)):
                index = i * dist.get_world_size() + rank + total
                # class_dir = os.path.join(sample_folder_classes_dir, f"class_{str(sample_label.item())}")
                # os.makedirs(class_dir, exist_ok=True)
                # Image.fromarray(sample).save(f"{class_dir}/{index:06d}.png")
                class_idx = str(sample_label.item())
                Image.fromarray(sample).save(os.path.join(sample_folder_classes_dir, f"{class_idx}_{index:06d}.png"))

        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        create_npz_from_sample_folder_with_classes(sample_folder_classes_dir)

        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode
    # Add interference to model
    parser.add_argument('--interference-path', type=str, default=None)
    parser.add_argument('--interference-weight', type=float, default=None)
    parser.add_argument('--velocity-weight', type=float, default=None)
    parser.add_argument('--reduce-interference', action=argparse.BooleanOptionalAction, default=False)


    args = parser.parse_args()
    main(args)