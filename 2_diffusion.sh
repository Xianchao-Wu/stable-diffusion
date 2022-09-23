#########################################################################
# File Name: 1_debug.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Sep  9 04:55:10 2022
#########################################################################
#!/bin/bash

# train auto encoder
# -t = for training
# --gpus = gpu index to be used for training
python -m ipdb main.py \
	--base "configs/latent-diffusion/cin-ldm-vq-f8.yaml" \
	-t \
	--gpus 0