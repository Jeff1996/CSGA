# CSGA
The implementation code of the paper "Enhancing Local Attention with Global Information Interaction via Progressive Cluster Propagation".

# Environment Configurations
Please refer to:

mmpretrain: https://github.com/open-mmlab/MMPreTrain?tab=readme-ov-file

mmsegmentation: https://github.com/open-mmlab/mmsegmentation

mmdetection: https://github.com/open-mmlab/mmdetection

Neighborhood-Attention-Transformer: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer

# 1. Classification
cd mmpretain

## Modify the configuration file
mmpretrain/configs/segformer/segformer_mit-b0_csga_4xb128_in1k-512x512.py

## Check the source code
mmpretrain/mmpretrain/models/backbones/mit_csga.py

## Train
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29531 bash tools/dist_train.sh path/to/the/config_file.py 4

## Test
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29532 bash tools/dist_test.sh path/to/the/config_file.py path/to/the/weights.pth 4

# 2. Segmentation
cd mmsegmentation

## Modify the configuration file
mmsegmentation/configs/segformer_ade20k/segformer_mit-b0_debug_2xb8-160k_ade20k-512x512.py

## Check the source code
mmsegmentation/mmseg/models/backbones/mit_csga.py

## Train
CUDA_VISIBLE_DEVICES=0,1 PORT=29531 bash tools/dist_train.sh path/to/the/config_file.py 2

## Test
CUDA_VISIBLE_DEVICES=0,1 PORT=29532 bash tools/dist_test.sh path/to/the/config_file.py path/to/the/weights.pth 2

# 3. Detection

##