# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import Upsample, resize
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class VQTHead(BaseDecodeHead):
    """
    量化聚类线性Transformer的辅助头，仅用于处理骨干网络中各个block的量化损失
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, inputs):
        """Forward function."""
        error_blocks = inputs[1]
        return error_blocks
