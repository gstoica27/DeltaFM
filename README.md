![DeltaFM Logo](assets/deltafm.png)

<h1 align="center"> Contrastive Flow Matching</h1>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2506.05350-b31b1b.svg)](https://arxiv.org/abs/2506.05350)&nbsp;

<div align="center">
  <a href="https://gstoica27.github.io" target="_blank">George&nbsp;Stoica</a><sup>12</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://vkramanuj.github.io" target="_blank">Vivek&nbsp;Ramanujan</a><sup>1*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://xiangfan.io" target="_blank">Xiang&nbsp;Fan</a><sup>1*</sup> &ensp; <b>
  <br>
  <a href="https://homes.cs.washington.edu/~ali/" target="_blank">Ali&nbsp;Farhadi</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://www.ranjaykrishna.com" target="_blank">Ranjay&nbsp;Krishna</a><sup>2</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://www.cc.gatech.edu/~judy/" target="_blank">Judy&nbsp;Hoffman</a><sup>1</sup><br>
  <sup>1</sup> Georgia Tech &emsp; <sup>2</sup>University of Washington &emsp; <br>
  <sup>*</sup>Equal Contribution &emsp; <br>
</div>
<h3 align="center">[<a href="https://arxiv.org/abs/2506.05350">arXiv</a>]</h3>

<b>Summary</b>: We propose Contrastive Flow Matching, a simple extension to the flow matching objective which reduces training iterations and denoising steps while improving generation quality.

### Awknowledgement
This repository is heavily built off of [REPA](https://github.com/sihyun-yu/REPA/tree/main) repository. Nearly all our files are the same as theirs, with the exception of a few scripts \& modifications. 


### 1. Environment setup
```bash
conda env create -f environment.yml
conda activate repa
```

### 2. Dataset
Please follow the setup described in [REPA](https://github.com/sihyun-yu/REPA/tree/main). We mirror their preprocessing procedures.

### 3. Training
Training with Contrastive Flow Matching is straightforward.
```bash
accelerate launch train.py \
  --report-to=wandb \
  --allow-tf32 \
  --mixed-precision=fp16 \
  --seed=0 \
  --path-type=linear \
  --prediction=v \
  --weighting=uniform \
  --enc-type=dinov2-vit-b \
  --output-dir=[CHECKPOINT_DIR] \
  --data-dir=[YOUR_DATA_PATH] \
  --model=SiT-XL/2 \
  --enc-type=dinov2-vit-b \
  --encoder-depth=8 \
  --batch-size 256 \
  --proj-coeff=0.5 \
  --loss-type "contrastive"  \ # Specify the contrastive objective. Use "mean" to specify the standard flow matching loss
  --contrastive-weight=0.05 \ # Specify the penalty (referred to as lambda in our work) to apply on the contrastive term
  --resolution 256
```
This script with automatically generate an experiment name based on the arguments passed, and save logs and checkpoints to the "output-dir". All remaining arguments match those of REPA. To run ImageNet512x512, please follow the procedure in REPA, and simply assign the "contrastive" argument to "--loss-type"

### 4. 
Our generation procedure is nearly equivalent to that of REPA, with a couple minor changes. Specifically, we alter the generation script to account for interference between CFG and Contrastive Flow Matching. 
To evaluate our models with CFG, please use the following script:
```bash
torchrun --nnodes=1 --nproc_per_node=8 --master-port 29501 generate.py \
  --model SiT-XL/2 \
  --num-fid-samples 50000 \
  --ckpt [MODEL_CHECKPOINT_PATH] \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=50 \
  --cfg-scale=1.85 \
  --guidance-high=0.65 \
  --sample-dir [SAMPLES_SAVE_DIR] \
  --interference-path /interference_vectors/imnet256_interference_vector.pt \ # path to computed interference vector following Section 5.4 of our paper
  --interference-weight 0.05 \ # Proportion of interference to remove from model
  --velocity-weight 0.95 \ # Rescaling weight for flow matching model
  --reduce-interference  # Apply interference reduction. Remove, to not apply it
```
To evaluate without CFG, you may use:
```bash
torchrun --nnodes=1 --nproc_per_node=8 --master-port 29501 generate.py \
  --model SiT-XL/2 \
  --num-fid-samples 50000 \
  --ckpt [MODEL_CHECKPOINT_PATH] \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=50 \
  --cfg-scale=1.0 \
  --sample-dir [SAMPLES_SAVE_DIR]
```
### BibTex
```bibtex
@misc{stoica2025contrastive,
  title={Contrastive Flow Matching}, 
  author={George Stoica and Vivek Ramanujan and Xiang Fan and Ali Farhadi and Ranjay Krishna and Judy Hoffman},
  year={2025},
  eprint={2506.05350},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.05350}, 
}
```

