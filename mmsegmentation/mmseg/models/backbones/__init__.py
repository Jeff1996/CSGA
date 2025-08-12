# Copyright (c) OpenMMLab. All rights reserved.
from .mit import MixVisionTransformer           # segformer
from .mit_csga import MixVisionTransformerMod   # segformer+csga

from .pvt import pvt_tiny, pvt_small, pvt_medium, pvt_large
                                                # pvt
from .pvt_csga import pvt_tinyMod, pvt_smallMod, pvt_mediumMod, pvt_largeMod
                                                # pvt+csga

from .nat import NAT                            # nat
from .nat_csga import NATMod                    # nat+csga

from .swin import SwinTransformer               # swin
from .swin_csga import SwinTransformerMod       # swin+csga
from .swin_cluster import SwinTransformerCluster# swin+ClusterFormer
from .swin_vq import SwinTransformerVQ          # swin+Transfor-VQ

from .twins import PCPVT, SVT                   # twins
from .twins_csga import PCPVTMod, SVTMod        # twins-csga

__all__ = [
    'MixVisionTransformer',
    'MixVisionTransformerMod',
    'pvt_tiny',
    'pvt_tinyMod',
    'NAT',
    'NATMod',
    'SwinTransformer',
    'SwinTransformerMod',
    'SwinTransformerCluster',
    'SwinTransformerVQ',
    'SVT',
    'SVTMod',
]
