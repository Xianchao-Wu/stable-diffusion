#########################################################################
# File Name: 3_txt2img_all.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Sep 29 04:09:13 2022
#########################################################################
#!/bin/bash

bash 3_txt2img_gpu1.sh > 3_txt2img_gpu1.sh.log.2 2>&1 &
bash 3_txt2img_gpu2.sh > 3_txt2img_gpu2.sh.log.2 2>&1 &
bash 3_txt2img_gpu3.sh > 3_txt2img_gpu3.sh.log.2 2>&1 &
bash 3_txt2img_gpu4.sh > 3_txt2img_gpu4.sh.log.2 2>&1 &
bash 3_txt2img_gpu5.sh > 3_txt2img_gpu5.sh.log.2 2>&1 &
bash 3_txt2img_gpu6.sh > 3_txt2img_gpu6.sh.log.2 2>&1 &
