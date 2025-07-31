# 测试距离矩阵构造函数
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool

import torch
import torch.nn.functional as F

# 设置随机数种子
def set_random_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)                           # 固定python的随机策略
    np.random.seed(seed)                        # 在使用 Numpy 库取随机数时，需要对其随机数种子进行限制
    torch.manual_seed(seed)                     # 当 Pytorch 使用 CPU 进行运算时，需要设定 CPU 支撑下的 Pytorch 随机数种子
    torch.cuda.manual_seed(seed)                # 单 GPU 情况
    torch.cuda.manual_seed_all(seed)            # 多 GPU 情况
    try:
        torch.backends.cudnn.benchmark = False  # 限制 Cudnn 在加速过程中涉及到的随机策略
        torch.backends.cudnn.deterministic = True
    except:
        pass


def getMask(size: tuple, index_onehot: torch.Tensor, gain: float=1.0):
    '''
    size: 特征图尺寸, [h, w]
    index_onehot: 聚类结果(每个像素对应的聚类中心的one-hot索引), [B, num_heads, L, S]
    gain: 增益系数
    '''
    assert type(size) == tuple, 'Data type of size in function <getMask> should be <tuple>!'
    assert size.__len__() == 2, 'Length of size should be 2!'
    coords_h = torch.arange(size[0])
    coords_w = torch.arange(size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))                          # 构造坐标窗口元素坐标索引，[2, h, w]
    # 一维化特征图像素坐标，[2, L]
    coords_featuremap = torch.flatten(coords, start_dim = 1).float().to(index_onehot.device)
    # [B, num_heads, 2, L]
    coords_featuremap = coords_featuremap.reshape(1, 1, 2, -1).repeat(index_onehot.shape[0], index_onehot.shape[1], 1, 1)
    # 聚类中心坐标，[B, num_heads, 2, S]
    coords_clustercenter = torch.einsum('bhcl,bhls->bhcs', coords_featuremap, index_onehot) / \
        torch.sum(index_onehot, dim=-2, keepdim=True)
    # 构造相对位置矩阵, 第一个矩阵是h方向的相对位置差, 第二个矩阵是w方向的相对位置差
    relative_coords = coords_featuremap[:, :, :, :, None] - coords_clustercenter[:, :, :, None, :]
    distance = torch.sqrt(                                                              # [B, num_heads, L, S]
        torch.square(relative_coords[:,:,0,:,:]) + torch.square(relative_coords[:,:,1,:,:])
    )
    distance_exp = torch.exp(distance)                                                  # exp操作用于处理distance中的0, [L, S]
    mask = (1 / distance_exp) * gain                                                    # 距离越远的token注意力增强越少(加性增强), 最大值为1*gain, 最小值可以接近0, [B, num_heads, L, S]
    return mask
    # return distance

# 获取初始相对位置编码(与初始化聚类中心构造方法有关), 如果图片尺寸不变化，则只需要构造一次
def initRPE(shape_x, shape_c, window_size, device):
    batch_size, num_heads, L_, _ = shape_c
    _, _, H, W, _ = shape_x
    H_, W_ = H // window_size[0], W // window_size[1]
    N = window_size[0] * window_size[1]

    delta_index = torch.arange(L_, device=device).view(1, 1, L_, 1).repeat(batch_size, num_heads, 1, N)
    delta_index = delta_index.reshape(
        batch_size, num_heads, H_, W_, window_size[0], window_size[1]
    ).permute(
        0, 1, 2, 4, 3, 5
    ).reshape(
        batch_size, num_heads, H_ * window_size[0], W_ * window_size[1]
    ).reshape(
        batch_size, num_heads, H*W
    )                                                           # [B, num_heads, L]
    delta_onehot = F.one_hot(delta_index, L_).float()           # [B, num_heads, L, L']
    relative_position_bias_init = getMask((H, W), delta_onehot) # [B, num_heads, L, L']
    return relative_position_bias_init

if __name__ == "__main__":
    set_random_seed(0)
    batch_size = 2
    num_heads = 2
    H, W = 4, 6
    head_embed_dims = 32
    window_size = (2, 2)
    L = H * W
    L_ = L // window_size[0] // window_size[1]

    size = (H, W)
    S = 3
    # index = torch.randint(0, S, (batch_size, num_heads, *size)).flatten(-2) # [B, num_heads, L]
    # print(index)
    # index_onehot = F.one_hot(index, S).float()                              # [B, num_heads, L, S]
    # print(index_onehot)
    # mask = getMask(size, index_onehot)
    # print(mask.shape)
    # print(mask)

    x = torch.randn((batch_size, num_heads, H, W, head_embed_dims), dtype=torch.float32, requires_grad=False)
    c = torch.randn((batch_size, num_heads, L_, head_embed_dims), dtype=torch.float32, requires_grad=False)
    relative_position_bias_init = initRPE(x.shape, c.shape, window_size, device='cpu')
    print('relative_position_bias_init: \n{} \n'.format(relative_position_bias_init))
    print('shape of relative_position_bias_init: \n{} \n'.format(relative_position_bias_init.shape))
