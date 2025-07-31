# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']                 # 按照原生的代码，这是一个列表，子元素直接就是torch.Tensor
        data_samples = data.get('data_samples', None)
                                                # 列表，子元素是mmseg.structures.seg_data_sample.SegDataSample

        if type(inputs[0]) == torch.Tensor:     # 常规图像分割数据
            # TODO: whether normalize should be after stack_batch
            if self.channel_conversion and inputs[0].size(0) == 3:
                inputs = [_input[[2, 1, 0], ...] for _input in inputs]

            inputs = [_input.float() for _input in inputs]
            if self._enable_normalize:
                inputs = [(_input - self.mean) / self.std for _input in inputs]

            if training:
                assert data_samples is not None, ('During training, ',
                                                '`data_samples` must be define.')
                inputs, data_samples = stack_batch(
                    inputs=inputs,
                    data_samples=data_samples,
                    size=self.size,
                    size_divisor=self.size_divisor,
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)

                if self.batch_augments is not None:
                    inputs, data_samples = self.batch_augments(
                        inputs, data_samples)
            else:
                img_size = inputs[0].shape[1:]
                assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                    'The image size in a batch should be the same.'
                # pad images when testing
                if self.test_cfg:
                    inputs, padded_samples = stack_batch(
                        inputs=inputs,
                        size=self.test_cfg.get('size', None),
                        size_divisor=self.test_cfg.get('size_divisor', None),
                        pad_val=self.pad_val,
                        seg_pad_val=self.seg_pad_val)
                    for data_sample, pad_info in zip(data_samples, padded_samples):
                        data_sample.set_metainfo({**pad_info})
                else:
                    inputs = torch.stack(inputs, dim=0)
            return dict(inputs=inputs, data_samples=data_samples)
        elif type(inputs[0]) == tuple:          # 少样本图像分割数据
            '''
            train阶段:
            列表的每个元素都是一个tuple, 一共3个元组, 每个元组包含一个batch的数据（跨task）
            第一个元组是一个batch的query data, 中间的元组是support data, 最后一个元组是异类样本(仅train阶段)
            所有元组同一位置的样本各自构成一个task

            val阶段:
            一共1+k个元组, 每个元组包含一个batch的数据（跨task）
            第一个元组是一个batch的query data, 剩下的k个元组是support data
            所有元组同一位置的样本各自构成一个task

            处理目标: 一个task的数据(query+support<+diff>)放在一起,
            一个task的inputs打包成[1+k<+1>, 3, H, W]的tensor, 不同task打包成[t, 1+k<+1>, 3, H, W]的tensor
            一个task的data_samples打包成一个列表task = [ds1, ds2, ...], 所有task再打包成一个列表batch = [task1, task2, ...]
            '''
            inputs_tasks = []
            data_samples_tasks = []
            for task in range(inputs[0].__len__()):
                                                # 获取任务数(query image的数量)
                inputs_task = [item[task] for item in inputs]
                                                # 获取当前任务中的所有图片
                
                data_samples_task = [item[task] for item in data_samples]
                                                # 获取当前任务中的所有标签及meta info

                if self.channel_conversion and inputs_task[0].size(0) == 3:
                    inputs_task = [item[[2, 1, 0], ...] for item in inputs_task]
                                                # 切换通道顺序
                inputs_task = [item.float() for item in inputs_task]
                if self._enable_normalize:
                    inputs_task = [(item - self.mean) / self.std for item in inputs_task]
                                                # 标准化
                
                # 由于验证过程也需要使用分割标签(s样本), 所以这里的无论模型是否处于训练状态, 都进行相同的填充处理
                inputs_task, data_samples_task = stack_batch(
                    inputs=inputs_task,
                    data_samples=data_samples_task,
                    size=self.size,         # 目标图片尺寸，如果经过pipeline处理的图片尺寸比这个尺寸小，则通过padding填充至此尺寸, 该值默认为512*512, 可在/home/hjf/workspace/mmsegmentation/configs/_base_/models/***.py下修改
                    size_divisor=self.size_divisor,
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)

                if self.batch_augments is not None:
                    inputs_task, data_samples_task = self.batch_augments(inputs_task, data_samples_task)

                inputs_tasks.append(inputs_task)
                data_samples_tasks.append(data_samples_task)
            return dict(inputs=inputs_tasks, data_samples=data_samples_tasks)
        else:
            raise NotImplementedError(
                "数据集构造出错, 请检查\n\
                文件: ./mmseg/datasets/basesegdataset.py, \n\
                类: BaseSegDataset, \n\
                方法: load_data_list\
                (该方法可能在子类中重构过./mmseg/datasets/basesegdataset.py)"
            )

