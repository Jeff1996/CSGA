# 少样本图像分割数据集PASCAL-5i
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

from typing import Callable, Dict, List, Optional, Sequence, Union
import mmengine
import os.path as osp
import logging
from mmengine.logging import print_log
import numpy as np
import torch

@DATASETS.register_module()
class PASCAL5iDataset(BaseSegDataset):
    """PASCAL-5i dataset.

    二分类任务，每个训练/验证任务需要从数据集中采样k个支持样本(含二值分割标签)和1个查询样本
    ，支持样本与查询样本来自同一个类别。
    背景索引为0, 前景索引为1
    """
    METAINFO = dict(
        classes=(
            'Background',
            'Foreground',
        ),
        palette=[
            [0, 0, 0],
            [255, 0, 0],
        ],
    )

    def __init__(
        self,
        folds: List[int] = [0],                 # 需要使用的fold, 例如[1, 2, 3]
        mode: str = 'train',                    # 使用fold*_train.txt还是fold*_val.txt
        shot: int = 1,                          # k-shot
        reduce_zero_label=False,                # 需要将背景类考虑进来
        **kwargs
    ) -> None:
        assert mode in ['train', 'val'], 'Undefined mode: {}'.format(mode)

        self.folds = folds
        self.mode = mode
        self.shot = shot
        self.data_cls_idxes = {}                # 按类划分的索引列表, 每个键值对包含了当前类的所有样本在self.data_list中的索引值
        self.data_idx_cls = []                  # 索引对应样本类别

        # for fold in folds:
        #     print('当前寻选择的数据集为:'+'fold{}_{}.txt'.format(fold, self.mode))
        # exit(0)

        super().__init__(
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )

    # 重写样本路径获取函数
    def load_data_list(self) -> List[dict]:
        '''
        PASCAL-5i等小样本数据集已经提前构造好了索引文件，索引文件中包含图片及其分割标签的相对路径
        如: JPEGImages/2008_000033.jpg Segmentations/01/2008_000033.png
        只需要拼接上所在目录的路径即可直接读取文件

        Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        '''

        data_list = []
        idx_global = 0                          # 对应data_list中的样本索引
        for fold in self.folds:
            path_idxfile = self.data_root + '/' + 'fold{}_{}.txt'.format(fold, self.mode)
            assert osp.isfile(path_idxfile), 'Failed to load ann_file: {}'.format(path_idxfile)

            lines = mmengine.list_from_file(path_idxfile, backend_args=self.backend_args)
            for line in lines:
                line = line.strip()
                path_img, path_anno = line.split(' ')
                                                # 相对路径
                path_img = self.data_root + '/' + path_img
                path_anno = self.data_root + '/' + path_anno

                data_info = dict(img_path=path_img)
                data_info['seg_map_path'] = path_anno
                data_info['label_map'] = self.label_map
                                                # 类别映射
                data_info['reduce_zero_label'] = self.reduce_zero_label
                                                # 是否忽略背景类
                data_info['seg_fields'] = []
                data_list.append(data_info)     # 保存数据信息

                # 增加的内容
                class_id = path_anno.split('/')[-2]
                                                # '01'、'02'……
                if not class_id in self.data_cls_idxes.keys():
                    self.data_cls_idxes[class_id] = [idx_global]
                else:
                    self.data_cls_idxes[class_id].append(idx_global)
                                                # 
                self.data_idx_cls.append(class_id)
                                                # 便于根据样本索引确定样本类别
                                                # 继而便于从同类样本中抽取支持图片
                idx_global += 1                 # 记录当前索引

        return data_list

    # 重写采样函数：输入一个idx, 返回一张query样本和shot张support样本以及一张异类样本（异类样本仅训练阶段需要）
    def __getitem__(self, idx: int) -> dict:
        """
        Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):

            # 获取查询样本
            data = self.prepare_data(idx)       # {'inputs': tensor, 'data_samples': SegDataSample}
            # Broken images or random augmentations may cause the returned data to be None
            if data is None:
                idx = self._rand_another()
                continue

            # 获取支持样本
            class_id = self.data_idx_cls[idx]   # 获取query样本的类别
            class_idxes = self.data_cls_idxes[class_id]
                                                # 获取query样本所在类别的全部样本的索引(含其本身)
            candidate_idxes = np.random.choice(class_idxes, class_idxes.__len__(), replace=False)
                                                # 打乱顺序
            support_idxes = candidate_idxes[~np.isin(candidate_idxes, idx)]
                                                # 排除query样本的索引
            num_support = 0
            support_datas = {'inputs': [data['inputs']], 'data_samples': [data['data_samples']]}
            for support_idx in support_idxes:
                support_data = self.prepare_data(support_idx)
                if support_data is None:
                    continue
                else:
                    num_support += 1
                    support_datas['inputs'].append(support_data['inputs'])
                    support_datas['data_samples'].append(support_data['data_samples'])
                if num_support >= self.shot:
                    break

            # 获取异类样本(仅训练过程需要)
            if self.mode == 'train':
                class_ids = np.array(self.data_idx_cls)
                class_ids_diff = class_ids[~np.isin(class_ids, class_id)]
                                                    # 获取异类ID
                
                for class_id_diff in np.random.choice(class_ids_diff, class_ids_diff.shape[0], replace=False):
                                                    # 随机选择一个异类
                    # 检查查询样本图片是否在当前选定的异类样本中（会影响查询样本作为s，异类样本作为q的分支训练的合理性）
                    if self.checkInclusiveness(idx, self.data_cls_idxes[class_id_diff]):
                        continue                    # 如果在，就得另选一个异类
                    else:
                        break
                
                class_idxes_diff = self.data_cls_idxes[class_id_diff]
                                                    # 获取当前异类的所有样本的索引值
                for class_idx_diff in np.random.choice(class_idxes_diff, class_idxes_diff.__len__(), replace=False):
                    # 检查当前异类样本图片是否在查询图片所在类别中（会影响查询样本作为q，异类样本作为s的分支训练的合理性）
                    if self.checkInclusiveness(class_idx_diff, class_idxes):
                        continue                    # 如果在，就得另选一个异类样本

                    diff_data = self.prepare_data(class_idx_diff)
                    if diff_data is None:
                        continue
                    else:
                        support_datas['inputs'].append(diff_data['inputs'])
                        support_datas['data_samples'].append(diff_data['data_samples'])
                        break

            # return data
            return support_datas

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')

    # 检查一个索引值代表的样本图片是否在一个索引值集合代表的样本图片中
    def checkInclusiveness(self, idx_q, idxes_k):
        '''
        idx_q: int, 待查询样本索引
        idxes_k: [int], 待查询样本索引集
        '''
        path_img_q = self.get_data_info(idx_q)['img_path']
        for idx_k in idxes_k:
            path_img_k = self.get_data_info(idx_k)['img_path']
            if path_img_q == path_img_k:
                return True
        return False
