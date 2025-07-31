# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet

from .mit import MixVisionTransformer           # 新增，来自mmpretrain
from .mit_debug import MixVisionTransformerMod  # 新增，来自mmpretrain

# from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
#                                                 # 原生
from .pvt import pvt_tiny, pvt_small, pvt_medium, pvt_large 
                                                # 新增，来自mmsegmentation
from .pvt_debug import pvt_tinyMod, pvt_smallMod, pvt_mediumMod, pvt_largeMod
                                                # 新增，来自mmsegmentation

from .nat import NAT                            # 新增，来自mmsegmentation
from .nat_debug import NATMod                   # 新增，来自mmsegmentation

from .twins import PCPVT, SVT                   # 新增，来自mmsegmentation
from .twins_debug import PCPVTMod, SVTMod       # 新增，来自mmsegmentation


__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'EfficientNet', 'CSPNeXt', 

    'MixVisionTransformer',                     # segformer
    'MixVisionTransformerMod',                  # segformer + csga

    'pvt_tiny',                                 # pvt
    'pvt_small', 
    'pvt_medium', 
    'pvt_large',

    'pvt_tinyMod',                              # pvt_debug
    'pvt_smallMod', 
    'pvt_mediumMod', 
    'pvt_largeMod',

    'NAT',                                      # nat
    'NATMod',                                   # nat_debug

    'SVT',                                      # twins-svt
    'SVTMod',                                   # twins-svt_debug
]
