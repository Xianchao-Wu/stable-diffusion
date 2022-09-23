#########################################################################
# File Name: 3_txt2img.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Sep  9 09:45:28 2022
#########################################################################
#!/bin/bash

python -m ipdb scripts/txt2img.py --plms \
	--ckpt "models/ldm/CompVis/stable-diffusion-v-1-4-original/stable-diffusion-v-1-4-original/sd-v1-4.ckpt"

# configs/stable-diffusion/v1-inference.yaml
