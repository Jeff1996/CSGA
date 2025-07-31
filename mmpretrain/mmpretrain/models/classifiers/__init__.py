# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .hugging_face import HuggingFaceClassifier
from .image import ImageClassifier
from .timm import TimmClassifier

from .image_vis import ImageClassifierVis
from .image_vq import ImageClassifierVQ
from .image_statistic import ImageClassifierSt

__all__ = [
    'BaseClassifier', 'ImageClassifier', 'TimmClassifier',
    'HuggingFaceClassifier', 'ImageClassifierVis', 'ImageClassifierVQ', 'ImageClassifierSt'
]
