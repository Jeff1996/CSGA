# k-means cluster, 基于k均值聚类的向量量化
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt

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

# 计算欧式距离
def getED(a: torch.Tensor, b: torch.Tensor):
    '''
    a: [L, d]
    b: [S, d]
    '''
    a2 = torch.sum(torch.square(a), dim=-1).unsqueeze(-1)   # [L, d] -> [L, ] -> [L, 1]
    ab = torch.einsum('ld,sd->ls', a, b)                    # [L, S]
    b2 = torch.sum(torch.square(b), dim=-1).unsqueeze(0)    # [S, d] -> [S, ] -> [1, S]
    distance2 = a2 - 2*ab + b2                              # a/b中每一个向量与码表中每一个向量的欧氏距离的平方, [L, S]
    distance = torch.sqrt(distance2)
    return distance

# 自定义梯度传播过程
class vecQuantization(Function):  
    @staticmethod  
    def forward(ctx, x, c):
        '''
        输入: 
        x           : tensor, [batch_size, heads, num_windows, N, dim], 分块的K矩阵
        c           : tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        输出:
        delta_onehot: tensor, [batch_size, heads, num_windows, N, num_windows], 量化的K矩阵
        c           : tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        # 前向传播（基于余弦相似度/单位化点积）
        cosSim = torch.einsum('bhrcd,bhsd->bhrcs', x, c)                    # 相似度矩阵, [batch_size, heads, num_windows, N, num_windows]
        delta_index = cosSim.argmax(dim=-1)                                 # 索引矩阵, [batch_size, heads, num_windows, N]
        delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()          # one-hot索引矩阵, [batch_size, heads, num_windows, N, num_windows]

        # # 前向传播（基于欧式距离）
        # x2 = torch.sum(torch.square(x), dim=-1).unsqueeze(-1)               # [B, heads, L, d] -> [B, heads, L] -> [B, heads, L, 1]
        # xc = torch.einsum('bhld,hsd->bhls', x, c)                           # [B, heads, L, S]
        # c2 = torch.sum(torch.square(c), dim=-1).unsqueeze(1).unsqueeze(0)   # [heads, S, d] -> [heads, S] -> [heads, 1, S] -> [1, heads, 1, S]
        # distance2 = x2 - 2*xc + c2                                          # 待量化序列中每一个向量与码表中每一个向量的欧氏距离的平方, [B, L, S]
        # delta_index = distance2.argmin(dim=-1)                              # 索引矩阵, [B, heads, L]
        # delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()           # one-hot索引矩阵, [B, heads, L, S]

        # print('------------ 调试点 ------------')
        # B, heads, L = delta_index.shape
        # num_tokens = B * heads * L
        # for index in torch.unique(delta_index):
        #     print('{}: {:.3f}%'.format(index, (delta_index==index).float().sum() / num_tokens * 100))

        # 保存必要的中间变量用于梯度反向传播
        ctx.save_for_backward(delta_onehot)

        return delta_onehot, c                                              # forward的输出个数与backward的输入个数相同
  
    @staticmethod  
    def backward(ctx, grad_output_delta_onehot, grad_output_c):  
        # 获取中间变量
        (delta_onehot, ) = ctx.saved_tensors
        # 梯度反传
        grad_x = torch.einsum('bhrcs,bhsd->bhrcd', delta_onehot, grad_output_c) # 来自码表c的梯度
        # backward的输出个数与forward的输入个数相同，如果某个输入变量不需要梯度，则对应返回None
        return grad_x, None                                        # backward的输出个数与forward的输入个数相同，如果某个输入变量不需要梯度，则对应返回None

# 向量量化器
class Quantizer(nn.Module):
    '''
    序列量化器
    1. 保存一个可训练的量化码表C
    2. 构造量化的K序列K^
    3. 获取输入矩阵K量化后的索引矩阵Δ
    '''
    def __init__(self, gamma: float=0.5, iterations: int=1):
        '''
        codes   : 码表中行向量的个数
        dim     : 码表每个行向量的维数(与K矩阵的维数一致)
        '''
        super(Quantizer, self).__init__()
        self.gamma = gamma                                      # EMA超参数（历史成分占比）
        self.iterations = iterations                            # 码表更新迭代次数
        self.vecQuantization = vecQuantization()                # 自定义梯度反传过程的向量量化函数

    def stopGradient(self, x: torch.tensor):
        '''
        梯度停止函数(stop gradient)
        '''
        return x.detach()
    
    def STE(self, value_forward: torch.tensor, grad_backward: torch.tensor):
        '''
        梯度直传函数(Straight-Through Estimator)
        解决由于argmin操作导致的梯度中断，
        前向传播时使用value_forward变量值，
        反向传播时grad_backward变量将继承value_forward变量的梯度

        输入：
        value_forward: 前向传播提供值，反向传播被继承梯度
        grad_backward: 前向传播无贡献，反向传播继承value_forward的梯度
        '''
        assert value_forward.shape == grad_backward.shape, "value_forward and grad_backward have different shapes!"
        return grad_backward + self.stopGradient(value_forward - grad_backward)

    # 使用类平均池化初始化聚类中心(码表)
    def initCodebook(self, x: torch.Tensor):
        '''
        输入: 
        x: tensor, [batch_size, heads, num_windows, N, dim], 分块的K矩阵
        输出:
        c: tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        c_init = torch.sum(x, dim=-2)                                           # [batch_size, heads, num_windows, dim], 将一个块内的元素求和
        c_init = c_init / torch.norm(c_init, dim=-1, keepdim=True)              # [batch_size, heads, num_windows, dim], 单位化码表
        return c_init

    # 迭代更新码表
    def updateCodebook(self, x: torch.Tensor, c: torch.Tensor):
        '''
        输入: 
        x: tensor, [batch_size, heads, num_windows*N, dim], 分块的K矩阵
        c: tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        输出：
        c: tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        affinity = torch.einsum('bhnd,bhld->bhnl', c, x)        # [batch_size, heads, num_windows, L], KC^T

        argmax_hw = affinity.argmax(dim=-1)                     # [batch_size, heads, num_windows]
        argmax_n = affinity.argmax(dim=-2)                      # [batch_size, heads, L]

        # [batch_size, heads, num_windows, L]
        argmax_hw_onehot = F.one_hot(argmax_hw, x.shape[-2]).float()
        # [batch_size, heads, num_windows, L]
        argmax_n_onehot = F.one_hot(argmax_n, c.shape[-2]).float().transpose(-2, -1)

        # attn = torch.softmax(attn, dim=-1) + torch.softmax(attn, dim=-2)
        # attn = torch.softmax(attn, dim=-1)
        attn = argmax_hw_onehot + argmax_n_onehot

        delta_c = torch.einsum('bhnl,bhld->bhnd', attn, x)   # [batch_size, heads, num_windows, dim]
        # delta_c = delta_c / torch.norm(delta_c, dim=-1, keepdim=True)
        c_new = self.gamma * c + (1 - self.gamma) * delta_c # 更新码表
        # c_new = delta_c
        c_new = c_new / torch.norm(c_new, dim=-1, keepdim=True)
        return c_new

    # 结合码表初始化与码表更新
    def getCodebook(self, x: torch.Tensor):
        '''
        输入: 
        x: tensor, [batch_size, heads, num_windows, N, dim], 分块的K矩阵
        输出:
        c: tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        # 初始化聚类中心
        c = self.initCodebook(x)                            # [batch_size, heads, num_windows, dim]
        # 通过k-means聚类更新码表c
        x = x.flatten(start_dim=2, end_dim=3)               # [batch_size, heads, L, dim]
        for _ in range(self.iterations):
            c = self.updateCodebook(x, c)
        return c

    def forward(self, x: torch.Tensor, batch_size: int=1):
        '''
        输入
        x           : tensor, [batch_size, heads, num_windows, N, dim], K矩阵
        输出
        delta_onehot: tensor, [batch_size, heads, num_windows, N, num_windows], 量化的K矩阵
        c           : tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        # 量化x
        delta_onehot, c = self.vecQuantization.apply(x, self.getCodebook(x))

        # print(delta_onehot)
        # print(c)

        return delta_onehot, c


if __name__ == "__main__":
    set_random_seed(0)
    path_root = '/home/hjf/workspace/mmsegmentation/work_dirs/swin-tiny-patch4-window7-LN_upernet_4xb2-80k_ade20k-512x512/'
    
    epoch_start = 10
    epochs = 10
    batch_size = 1
    num_windows = 100
    heads = 3
    N = 49
    dim = 32

    quantizer = Quantizer(
        gamma=0.5,      # 码表迭代更新过程的EMA参数
        iterations=0    # 码表更新迭代次数, 0此表示不迭代，直接使用类池化结果作为码表
    )

    datas1 = torch.randn(batch_size*num_windows, heads, N, dim).abs()
    datas2 = torch.randn(batch_size*num_windows, heads, N, dim).abs() * -1
    datas = torch.concat([datas1, datas2], dim=0)
    datas = datas / torch.norm(datas, dim=-1, keepdim=True)             # 将数据缩放至单位球

    with torch.no_grad():
        B, heads, N, dim = datas.shape
        num_windows = B // batch_size
        # [batch_size, heads, num_windows, N, dim]
        datas = datas.view(batch_size, num_windows, heads, N, dim).permute(0, 2, 1, 3, 4).contiguous()

        # print(datas.shape)
        # print(datas)

        # for epoch in tqdm(torch.arange(1, epochs+1)):
        for epoch in torch.arange(epoch_start, epoch_start+epochs):
            quantizer.iterations = epoch - 1

            # [batch_size, heads, num_windows, N, num_windows], [batch_size, heads, num_windows, dim]
            delta_onehot, c = quantizer(datas)

            # 计算量化误差
            datas_hat = torch.einsum('bhrcs,bhsd->bhrcd', delta_onehot, c)
            error_quantization = torch.norm(datas.detach() - datas_hat.detach(), dim=-1).square().mean()
            print('码表更新次数: {:2d}, 量化误差: {:.6f}'.format(quantizer.iterations, error_quantization.item()))

            # --- 可视化码表更新过程 ---
            # 中心化、单位化
            x = datas
            # x = datas - datas.mean(dim=(0, -2), keepdim=True)
            # x = x / torch.norm(x, dim=-1, keepdim=True)
            x = x.view(-1, dim).numpy()                                         # [B*heads*tokens, dim]
            codebook = c
            # codebook = c - c.mean(dim=-2, keepdim=True)
            # codebook = codebook / torch.norm(codebook, dim=-1, keepdim=True)
            codebook = codebook.view(-1, dim).numpy()                           # [heads*codes, dim]
            # 可视化
            path_scatter = path_root + 'codebook_update_epoch_{:02d}.png'.format(epoch)
            plt.figure(figsize=(10, 10))
            colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
            plt.scatter(x[:, 0], x[:, 1], c='g', label='datas')
            plt.scatter(codebook[:, 0], codebook[:, 1], c='r', label='codebook')
            plt.legend()
            plt.savefig(path_scatter, dpi=300)
            plt.clf()
            plt.close()

    #         datas_hat = torch.einsum('bhls,hsk->bhlk', delta_onehot, c)
    #         error_quantization = torch.norm(datas.detach() - datas_hat.detach(), dim=-1).square().mean()
    #         print('余弦相似度量化损失:', error_quantization)

    #     error_quantization = torch.norm(datas.detach() - datas.detach().mean(dim=-2), dim=-1).square().mean()
    #     print('均值量化损失：', error_quantization)

    #     # # --- 可视化自注意力 ---
    #     # attn = torch.softmax(torch.einsum('bhld,bhnd->bhln', datas, datas), dim=-1).view(tokens, tokens)
    #     # attn_hat = torch.softmax(torch.einsum('bhld,bhnd->bhln', datas, datas_hat), dim=-1).view(tokens, tokens)
    #     # # # 可视化注意力图
    #     # GradVis(attn.cpu().detach(), 'attn', path_root + 'attention_map.png')
    #     # GradVis(attn_hat.cpu().detach(), 'attn_hat', path_root + 'attention_map_quantization.png')
    #     # # 可视化单个token的注意力
    #     # index_token = 0
    #     # x = torch.arange(1, attn.shape[-1]+1)
    #     # path_hist = path_root + 'softmax_of_token_{:04d}.png'.format(index_token)
    #     # plt.bar(x, attn[index_token].cpu().numpy(), color='skyblue')  
    #     # plt.title('References Distribution')
    #     # plt.xlabel('code index')
    #     # plt.ylabel('Frequency')
    #     # plt.grid(True)
    #     # plt.savefig(path_hist, dpi=300)
    #     # plt.clf()
    #     # plt.close()
    #     # path_hist = path_root + 'softmax_of_quantization_token_{:04d}.png'.format(index_token)
    #     # plt.bar(x, attn_hat[index_token].cpu().numpy(), color='skyblue')  
    #     # plt.title('References Distribution')
    #     # plt.xlabel('code index')
    #     # plt.ylabel('Frequency')
    #     # plt.grid(True)
    #     # plt.savefig(path_hist, dpi=300)
    #     # plt.clf()
    #     # plt.close()

    #     # # print(quantizer.c.data)
    #     # print('码表中更新次数最多的向量的均值：', quantizer.c[:, quantizer.c_count.argmax(dim=-1)].mean())
    #     # print('数据的均值：', datas.mean())

    #     # 可视化码表向量更新次数
    #     num_update = quantizer.c_count.squeeze().detach().cpu().sort()[0]
    #     x = np.arange(num_update.shape[0])
    #     path_hist = path_root + 'num_update.png'
    #     plt.bar(x, num_update.cpu().numpy(), color='skyblue')  
    #     plt.title('update times of vectors in codebook')
    #     plt.xlabel('code index')
    #     plt.ylabel('Frequency')
    #     plt.grid(True)
    #     plt.savefig(path_hist, dpi=300)
    #     plt.close()
