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
# --base "configs/autoencoder/autoencoder_kl_64x64x3.yaml" # NOTE TODO this is the first time used! \
# naotu was okay: 
# https://naotu.baidu.com/file/ad2c8379fc533d8bc743c52851353f79
# https://naotu.baidu.com/file/44c5f9005c37f3392529c8a79cc2b195

python -m ipdb main.py \
	--base "configs/autoencoder/autoencoder_kl_32x32x4.yaml" \
	-t \
	--gpus 0
