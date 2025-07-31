# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .ddrnet import DDRNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mit_debug import MixVisionTransformerMod              # 新增
from .mit_debug2 import MixVisionTransformerMod2            # 新增

from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mscan import MSCAN
from .pidnet import PIDNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone

from .twins import PCPVT, SVT
from .twins_debug import PCPVTMod, SVTMod                   # 新增
from .twins_debug2 import PCPVTMod2, SVTMod2                # 新增

from .unet import UNet
from .vit import VisionTransformer
from .vit_debug import VisionTransformerMod
from .vpd import VPD

from .vqt import VQTransformer                              # 新增
from .vqswin_ema import VQSwinTransformer                   # 新增
from .swin_debug import SwinTransformerMod                  # 新增
from .swin_debug2 import SwinTransformerMod2                # 新增
from .swin_debug_statistic import SwinTransformerModSt      # 新增，统计码表利用率

from .nat import NAT                                        # 新增
from .nat_debug import NATMod                               # 新增
from .nat_debug2 import NATMod2                             # 新增
from .pvt import pvt_tiny, pvt_small, pvt_medium, pvt_large # 新增
from .pvt_debug import pvt_tinyMod, pvt_smallMod, pvt_mediumMod, pvt_largeMod # 新增
from .pvt_debug2 import pvt_tinyMod2, pvt_smallMod2, pvt_mediumMod2, pvt_largeMod2 # 新增
from .swin_cluster import SwinTransformerCluster            # 新增
from .swin_vq import SwinTransformerVQ                      # 新增
from .swin_vq_statistic import SwinTransformerVQSt          # 新增，统计码表利用率

from .nat_fss import NATFSS                                 # 基于NAT+CSGA的少样本图像分割框架
from .swin_dcama import SwinTransformerDCAMA                # 用于DCAMA少样本分割模型的骨干网络

__all__ = [
    'ResNet', 
    'ResNetV1c', 
    'ResNetV1d', 
    'ResNeXt', 
    'HRNet', 
    'FastSCNN',
    'ResNeSt', 
    'MobileNetV2', 
    'UNet', 
    'CGNet', 
    'MobileNetV3',
    'VisionTransformer', 
    'SwinTransformer', 
    'MixVisionTransformer',
    'BiSeNetV1', 
    'BiSeNetV2', 
    'ICNet', 
    'TIMMBackbone', 
    'ERFNet', 
    'PCPVT',
    'SVT', 
    'STDCNet', 
    'STDCContextPathNet', 
    'BEiT', 
    'MAE', 
    'PIDNet', 
    'MSCAN',
    'DDRNet', 
    'VPD', 
    'VQTransformer', 
    'VQSwinTransformer', 

    'VisionTransformerMod',     # 01 vit_debug
                                # 02 setr_debug(直接用vit_debug)

    'MixVisionTransformerMod',  # 03 segformer_debug
    'MixVisionTransformerMod2', # 03 segformer_debug2

    'SwinTransformerMod',       # 04 swin_debug
    'SwinTransformerMod2',      # 04 swin_debug2
    'SwinTransformerModSt',     # 04 swin_debug, 统计码表利用率
    
    'pvt_tiny',                 # 05 pvt
    'pvt_small', 
    'pvt_medium', 
    'pvt_large',

    'pvt_tinyMod',              # 05 pvt_debug
    'pvt_smallMod', 
    'pvt_mediumMod', 
    'pvt_largeMod',

    'pvt_tinyMod2',              # 05 pvt_debug2
    'pvt_smallMod2', 
    'pvt_mediumMod2', 
    'pvt_largeMod2',

    'NAT',                      # 06 nat
    'NATMod',                   # 06 nat_debug
    'NATMod2',                  # 06 nat_debug2

    'SVTMod',                   # 07 twins_debug
    'SVTMod2',                  # 07 twins_debug, 去除相对位置编码

    'SwinTransformerCluster',   # 08 Swin+ClusterFormer
    'SwinTransformerVQ',        # 09 Swin+VQ
    'SwinTransformerVQSt',      # 09 Swin+VQ, 统计码表利用率

    'NATFSS',                                   # 基于NAT+CSGA的少样本图像分割框架
    'SwinTransformerDCAMA',                     # 用于DCAMA少样本分割模型的骨干网络
]
