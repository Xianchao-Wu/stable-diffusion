#########################################################################
# File Name: 1_debug.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Sep  9 04:55:10 2022
#########################################################################
#!/bin/bash

# train auto encoder
# -t = for training
# --gpus = num of gpus to be used for training
# --base "configs/autoencoder/autoencoder_kl_64x64x3.yaml" # NOTE TODO this is the first time used! \
# naotu was okay: 
# https://naotu.baidu.com/file/ad2c8379fc533d8bc743c52851353f79
# https://naotu.baidu.com/file/44c5f9005c37f3392529c8a79cc2b195

python -m ipdb main.py \
	--base "configs/autoencoder/autoencoder_kl_32x32x4.yaml" \
	-t \
	--gpus "1" # 0 = use 0 does not work!

# NOTE TODO 需要注意的是，先是要搞一下validation，
# 所以，一开始的几步的forward里面，张量是没有梯度相关的信息的！！
# 这一点一定要注意，不要上当了.

# 等validation结束之后，真正train的时候，才会有梯度函数信息，被
# 记录到各个张量上面。
