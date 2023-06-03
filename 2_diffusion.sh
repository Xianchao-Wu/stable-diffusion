#########################################################################
# File Name: 2_diffusion.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Sep  9 04:55:10 2022
#########################################################################
#!/bin/bash

# -t = for training
# --gpus = num of gpus (index="1,2,3...,8") to be used for training
python -m ipdb main.py \
	--base "configs/latent-diffusion/cin-ldm-vq-f8.yaml" \
	-t \
	--gpus '1' #0
