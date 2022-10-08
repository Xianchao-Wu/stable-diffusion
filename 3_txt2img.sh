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

hw=1024
#infn2="prompts2.txt"
#infn3="prompts3.txt"
infn3="prompts_txt/prompts23.txt"

#outdir1="outdir1" # ddim_steps=50, seed=42
#outdir2="outdir2" # ddim_steps=200, seed=888
outdir3="outdir23_$hw" # ddim_steps=200, seed=42, reuse "stable diffusion" paper's prompts and 30 artists

python scripts/txt2img_select_gpu.py \
	--from-file $infn3 \
	--outdir $outdir3 \
	--ddim_eta 1.0 \
	--n_samples 1 \
	--n_iter 5 \
	--ddim_steps 200 \
	--H $hw \
	--W $hw \
	--scale 5.0 \
	--seed 42 \
	--device "cuda:0" \
	--config "configs/stable-diffusion/v1-inference-gpu0.yaml" \
	--ckpt "models/ldm/CompVis/stable-diffusion-v-1-4-original/stable-diffusion-v-1-4-original/sd-v1-4.ckpt"

