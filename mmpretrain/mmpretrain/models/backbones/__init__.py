# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .beit import BEiTViT
from .conformer import Conformer
from .convmixer import ConvMixer
from .convnext import ConvNeXt
from .cspnet import CSPDarkNet, CSPNet, CSPResNet, CSPResNeXt
from .davit import DaViT
from .deit import DistilledVisionTransformer
from .deit3 import DeiT3
from .densenet import DenseNet
from .edgenext import EdgeNeXt
from .efficientformer import EfficientFormer
from .efficientnet import EfficientNet
from .efficientnet_v2 import EfficientNetV2
from .hivit import HiViT
from .hornet import HorNet
from .hrnet import HRNet
from .inception_v3 import InceptionV3
from .lenet import LeNet5
from .levit import LeViT
from .mixmim import MixMIMTransformer
from .mlp_mixer import MlpMixer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mobileone import MobileOne
from .mobilevit import MobileViT
from .mvit import MViT
from .poolformer import PoolFormer
from .regnet import RegNet
from .replknet import RepLKNet
from .repmlp import RepMLPNet
from .repvgg import RepVGG
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .resnext import ResNeXt
from .revvit import RevVisionTransformer
from .riformer import RIFormer
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .sparse_convnext import SparseConvNeXt
from .sparse_resnet import SparseResNet

from .swin_transformer import SwinTransformer
from .swin_debug import  SwinTransformerMod                 # 新增
from .swin_debug2 import  SwinTransformerMod2               # 新增
from .swin_debug_statistic import SwinTransformerModSt      # 新增，用于Swin+CSGA统计码表利用率

from .swin_transformer_v2 import SwinTransformerV2
from .t2t_vit import T2T_ViT
from .timm_backbone import TIMMBackbone
from .tinyvit import TinyViT
from .tnt import TNT
from .twins import PCPVT, SVT
from .twins_debug import PCPVTMod, SVTMod                   # 新增
from .twins_debug2 import PCPVTMod2, SVTMod2                   # 新增

from .van import VAN
from .vgg import VGG
from .vig import PyramidVig, Vig

from .vision_transformer import VisionTransformer
from .vision_transformer_debug import VisionTransformerMod  # 新增

from .vit_eva02 import ViTEVA02
from .vit_sam import ViTSAM
from .xcit import XCiT

from .mit import MixVisionTransformer                       # 新增
from .mit_debug import MixVisionTransformerMod              # 新增
from .mit_debug2 import MixVisionTransformerMod2            # 新增

from .nat import NAT                                        # 新增
from .nat_debug import NATMod                               # 新增
from .nat_debug_vis import NATModVis                        # 新增
from .nat_debug2 import NATMod2                             # 新增
from .pvt import pvt_tiny, pvt_small, pvt_medium, pvt_large # 新增
from .pvt_debug import pvt_tinyMod, pvt_smallMod, pvt_mediumMod, pvt_largeMod       # 新增
from .pvt_debug2 import pvt_tinyMod2, pvt_smallMod2, pvt_mediumMod2, pvt_largeMod2   # 新增

from .cluster import cluster_tiny, cluster_small            # 新增
from .cluster_debug import cluster_tinyMod, cluster_smallMod# 新增
from .swin_cluster import SwinTransformerCluster            # 新增
from .swin_vq import SwinTransformerVQ                      # 新增
from .swin_vq_statistic import SwinTransformerVQSt          # 新增，用于Transformer-VQ的统计码表利用率
from .swin_debug_patent import SwinTransformerPatent        # 新增，用于专利申请

__all__ = [
    'LeNet5',
    'AlexNet',
    'VGG',
    'RegNet',
    'ResNet',
    'ResNeXt',
    'ResNetV1d',
    'ResNeSt',
    'ResNet_CIFAR',
    'SEResNet',
    'SEResNeXt',
    'ShuffleNetV1',
    'ShuffleNetV2',
    'MobileNetV2',
    'MobileNetV3',
    'VisionTransformer',            # 原生01
    'SwinTransformer',              # 原生04
    'TNT',
    'TIMMBackbone',
    'T2T_ViT',
    'Res2Net',
    'RepVGG',
    'Conformer',
    'MlpMixer',
    'DistilledVisionTransformer',
    'PCPVT',
    'SVT',
    'EfficientNet',
    'EfficientNetV2',
    'ConvNeXt',
    'HRNet',
    'ResNetV1c',
    'ConvMixer',
    'EdgeNeXt',
    'CSPDarkNet',
    'CSPResNet',
    'CSPResNeXt',
    'CSPNet',
    'RepLKNet',
    'RepMLPNet',
    'PoolFormer',
    'RIFormer',
    'DenseNet',
    'VAN',
    'InceptionV3',
    'MobileOne',
    'EfficientFormer',
    'SwinTransformerV2',
    'MViT',
    'DeiT3',
    'HorNet',
    'MobileViT',
    'DaViT',
    'BEiTViT',
    'RevVisionTransformer',
    'MixMIMTransformer',
    'TinyViT',
    'LeViT',
    'Vig',
    'PyramidVig',
    'XCiT',
    'ViTSAM',
    'ViTEVA02',
    'HiViT',
    'SparseResNet',
    'SparseConvNeXt',
    'VisionTransformerMod',         # 01 vit_debug
                                    # 02 setr直接用vit和vit_debug

    'MixVisionTransformer',         # 03 segformer
    'MixVisionTransformerMod',      # 03 segformer_debug
    'MixVisionTransformerMod2',     # 03 segformer_debug, 去掉聚类过程的相对位置编码

    'SwinTransformerMod',           # 04 swin_debug
    'SwinTransformerMod2',          # 04 swin_debug, 去掉聚类过程的相对位置编码
    'SwinTransformerModSt',         # 04 swin_debug, 统计码表利用率

    'pvt_tiny',                     # 05 pvt
    'pvt_small', 
    'pvt_medium', 
    'pvt_large',

    'pvt_tinyMod',                  # 05 pvt_debug
    'pvt_smallMod', 
    'pvt_mediumMod', 
    'pvt_largeMod',

    'pvt_tinyMod2',                 # 05 pvt_debug, 去掉聚类过程的相对位置编码
    'pvt_smallMod2', 
    'pvt_mediumMod2', 
    'pvt_largeMod2',

    'NAT',                          # 06 nat
    'NATMod',                       # 06 nat_debug
    'NATModVis',                    # 06 nat_debug, 用于可视化中间过程的代码
    'NATMod2',                      # 06 nat_debug, 去掉聚类过程的相对位置编码

    'SVTMod',                       # 07 twins-svt_debug
    'SVTMod2',                      # 07 twins-svt_debug, 去掉聚类过程的相对位置编码

    'cluster_tiny',                 # 08 ClusterFormer
    'cluster_tinyMod',              # 08 ClusterFormer_debug
    'SwinTransformerCluster',       # 08 Swin+ClusterFormer
    'SwinTransformerVQ',            # 09 Swin+VQ
    'SwinTransformerVQSt',          # 09 Swin+VQ, 统计码表利用率
    'SwinTransformerPatent',        # 新增，用于专利申请
]
