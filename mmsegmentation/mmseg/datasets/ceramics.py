# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CeramicsDataset(BaseSegDataset):
    """Ceramics dataset.

    In segmentation map annotation for Ceramics, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=(
            'C01',
            'C04',
            'C06',
            'C08',
            'P23',
            'T52_2',
        ),
        palette=[
            [217, 213, 180],
            [218, 147, 70],
            [234, 132, 163],
            [61, 127, 236],
            [81, 202, 147],
            [211, 109, 209],
        ]
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
