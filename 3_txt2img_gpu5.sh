#########################################################################
# File Name: 3_txt2img.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Sep  9 09:45:28 2022
#########################################################################
#!/bin/bash

# debug only
#python -m ipdb scripts/txt2img.py --plms \
#	--ckpt "models/ldm/CompVis/stable-diffusion-v-1-4-original/stable-diffusion-v-1-4-original/sd-v1-4.ckpt"

# configs/stable-diffusion/v1-inference.yaml
#--prompt "a painting of last dinner by vincent van gogh" \

#infn2="prompts2.txt"
#infn3="prompts3.txt"
infn="x04" # TODO

#outdir1="outdir1" # ddim_steps=50, seed=42
#outdir2="outdir2" # ddim_steps=200, seed=888
outdir="outdir_x04" # TODO ddim_steps=200, seed=42, reuse "stable diffusion" paper's prompts and 30 artists

# TODO device
# TODO yaml

python scripts/txt2img_select_gpu.py \
	--from-file $infn \
	--outdir $outdir \
	--ddim_eta 1.0 \
	--n_samples 5 \
	--n_iter 5 \
	--ddim_steps 200 \
	--H 512 \
	--W 512 \
	--scale 5.0 \
	--seed 42 \
	--device "cuda:5" \
	--config "configs/stable-diffusion/v1-inference-gpu5.yaml" \
	--ckpt "models/ldm/CompVis/stable-diffusion-v-1-4-original/stable-diffusion-v-1-4-original/sd-v1-4.ckpt"

