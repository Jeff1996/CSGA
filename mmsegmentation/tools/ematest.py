# 测试程序，使用EMA方法，初始化一组聚类中心，基于欧氏距离对一组满足高斯分布的数据点进行迭代聚类
import numpy as np
import os
import random
from tqdm import tqdm
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

# 热图可视化
def GradVis(grad: torch.Tensor, title: str='', path_save: str=''):
    '''
    grad: [H, W]
    '''
    assert grad.ndim == 2, "只能接受二维矩阵，实际收到的矩阵形状为: {}".format(grad.shape)

    # 创建一个图形和轴
    fig, ax = plt.subplots()
    
    # 使用imshow绘制热力图
    # 第一个参数是数据矩阵，第二个参数是颜色映射（可选参数包括： viridis, hot, cool, jet）
    cax = ax.imshow(grad, cmap='viridis', interpolation='nearest')
    
    # 添加颜色条
    fig.colorbar(cax)
    
    # 添加标签
    ax.set_xlabel('W')
    ax.set_ylabel('H')
    title = 'Heatmap of grad' if title == '' else title
    ax.set_title(title)

    if path_save == '': # 显示图片
        plt.show()
    else:               # 保存图片
        plt.savefig(path_save, dpi=300)
    plt.clf()
    plt.close
    

# 自定义梯度传播过程
class vecQuantization(Function):  
    @staticmethod  
    def forward(ctx, x, c):
        # 前向传播（基于余弦相似度/单位化点积）
        # x = x - x.mean(dim=(0, 2), keepdim=True)                            # 这里可以改为EMA更新的均值
        # x = x / torch.norm(x, dim=-1, keepdim=True)
        # codebook = c - c.mean(dim=-2, keepdim=True)
        # codebook = codebook / torch.norm(codebook, dim=-1, keepdim=True)
        cosSim = torch.einsum('bhld,hsd->bhls', x, c)                       # 相似度矩阵, [B, heads, L, S]
        delta_index = cosSim.argmax(dim=-1)                                 # 索引矩阵, [B, heads, L]
        delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()          # one-hot索引矩阵, [B, heads, L, S]

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
        grad_x = torch.einsum('bhls,hsd->bhld', delta_onehot, grad_output_c)# 来自码表c的梯度
        return grad_x, None                                                 # backward的输出个数与forward的输入个数相同，如果某个输入变量不需要梯度，则对应返回None

# 向量量化器
class Quantizer(nn.Module):
    '''
    序列量化器
    1. 保存一个可训练的量化码表C
    2. 构造量化的K序列K^
    3. 获取输入矩阵K量化后的索引矩阵Δ
    '''
    def __init__(self, heads: int, codes: int, dim: int, ema: bool = True):
        '''
        codes   : 码表中行向量的个数
        dim     : 码表每个行向量的维数(与K矩阵的维数一致)
        '''
        super(Quantizer, self).__init__()
        self.ema = ema                                                          # 是否采用EMA更新码表
        c_init = torch.randn(heads, codes, dim)
        c_init = c_init / torch.norm(c_init, dim=-1, keepdim=True)              # 单位化的码表

        if ema:
            # 使用EMA更新的码表
            self.c = nn.Parameter(c_init, requires_grad=False) # 汇总的码表
            # # 也可以考虑使用register_buffer方法定义上述两个不需要梯度更新的参数
            # self.register_buffer('c', c_init)
            c_sum_new = torch.zeros(heads, codes, dim)                          # 用于累计更新量
            self.c_sum_new = nn.Parameter(c_sum_new, requires_grad=False)
            c_count_new = torch.zeros(heads, codes)                             # 用于归一化更新量
            self.c_count_new = nn.Parameter(c_count_new, requires_grad=False)
            self.c_count = nn.Parameter(c_count_new, requires_grad=False)       # 用于统计整个更新过程，码表中的每个向量被更新的次数

            self.update_count = 1                                               # 更新量累计次数统计
            self.update_interval = 5                                            # 码表更新间隔(码表两次更新间的更新量累计次数)
        else:
            # 使用梯度更新的码表
            self.c = nn.Parameter(c_init)

        self.gamma = 0.9                                                        # EMA超参数（历史成分占比）
        self.vecQuantization = vecQuantization()                                # 自定义梯度反传过程的向量量化函数

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

    def getCodebook(self):
        '''
        返回归一化的码表
        c: [heads, codes, dim]
        '''
        c = self.c.data                                                     # 获取码表
        return c

    def emaAccumulate(self, delta_onehot, x):
        '''
        累计码表更新量
        更新量累计在量化之后，码表更新在量化之前，由此才不会影响梯度反向传播
        '''
        c_sum_new = torch.einsum('bhls,bhld->hsd', delta_onehot, x)         # [heads, S, dim]
        c_count_new = torch.einsum('bhls->hs', delta_onehot)                # [heads, S]
        self.c_sum_new.data = self.c_sum_new.data + c_sum_new.detach()              # 累计
        self.c_count_new.data = self.c_count_new.data + c_count_new.detach()        # 累计
        self.c_count.data = self.c_count.data + c_count_new.detach()                # 测试用参数，不清零

    def updateCodebook(self):
        '''
        EMA更新码表
        delta_onehot: tensor, [batch size, heads, L, S], 量化K的one-hot索引矩阵Δ
        x           : tensor, [batch size, heads, L, dim], K矩阵
        '''
        # 计算更新量（单位化）
        c_sum_new = self.c_sum_new.data                                     # [heads, S, dim]
        c_count_new = self.c_count_new.data.unsqueeze(-1)                   # [heads, S, 1]
        c_count_new[c_count_new==0] = 1                                     # 防止除以0
        c_new = c_sum_new / c_count_new                                     # 计算平均更新量
        c_new_norm = torch.norm(c_new, dim=-1, keepdim=True)                # 计算码表更新量中每个向量的二范数，用于单位化
        c_new_norm[c_new_norm==0] = 1                                       # 防止除以0
        c_new = c_new / c_new_norm                                          # 单位化更新量
        c = self.gamma * self.c + (1 - self.gamma) * c_new                  # EMA更新码表
        c_norm = torch.norm(c, dim=-1, keepdim=True)                        # 计算更新后码表中每个向量的二范数，用于单位化
        c_norm[c_norm==0] = 1                                               # 防止除以0
        c = c / c_norm                                                      # 单位化码表

        # # 计算更新量（非单位化，配置基于欧氏距离的向量量化方案）
        # c_sum_new = self.c_sum_new.data                                     # [heads, S, dim]
        # index_notzero = self.c_count_new.data > 0                           # 用于决定更新哪些码表向量
        # c_count_new = self.c_count_new.data.unsqueeze(-1)                   # [heads, S, 1]
        # c_count_new[c_count_new==0] = 1                                     # 防止除以0
        # c_new = c_sum_new / c_count_new                                     # 计算平均更新量
        # c = self.c.data
        # c[index_notzero] = self.gamma * c[index_notzero] + (1 - self.gamma) * c_new[index_notzero]  # EMA更新码表

        # 对于更新量最小的码表向量(没有更新的)，则对将其向更新量最大的向量偏移
        index_dense = F.one_hot(self.c_count_new.argmax(dim=-1), c.shape[-2]).float()   # 找到每个head中更新次数最多的向量
        vectors_dense = torch.einsum('hsd,hs->hd', c, index_dense)          # [heads, dim]
        for index_head, c_count_new in enumerate(self.c_count_new.data):
            if torch.any(c_count_new <= 1):
                index_sparse = c_count_new <= 1
            else:
                index_sparse = c_count_new.argmin(dim=-1)
            c[index_head][index_sparse] = c[index_head][index_sparse] + vectors_dense[index_head]
            c[index_head][index_sparse] = c[index_head][index_sparse] / torch.norm(c[index_head][index_sparse], dim=-1, keepdim=True)

        # 更新码表
        self.c.data = c                                                     # 更新码表
        self.c_sum_new.data.fill_(.0)                                       # 累积量清零
        self.c_count_new.data.fill_(.0)                                     # 累积量清零

    def forward(self, x):
        '''
        输入
        x               : tensor, [batch size, heads, L, dim], K矩阵

        输出
        delta_onehot    : tensor, [batch size, heads, L , S], 量化K的索引矩阵Δ
        c               : tensor, [heads, S, dim], 量化码表
        '''
        # 更新码表(避免影响本次梯度反向传播, 在量化操作前进行更新)
        if self.update_count % self.update_interval == 0:
            self.updateCodebook()

        # 量化(需要将c矩阵获取到的梯度中继给x, delta_onehot的梯度则停掉)
        delta_onehot, c = self.vecQuantization.apply(x, self.getCodebook())

        # 累计码表更新量(量化操作之后进行)
        self.update_count += 1
        self.emaAccumulate(delta_onehot, x)

        return delta_onehot, c

if __name__ == "__main__":
    set_random_seed(0)
    path_root = '/home/student/hjf/workspace/mmsegmentation/work_dirs/swin-tiny-patch4-window7-LN_upernet_2xb1-80k_ade20k-512x512/'
    epochs = 100
    batchsize = 1
    heads = 1
    codes = 128
    dim = 2
    tokens = 128**2
    quantizer = Quantizer(
        heads=heads, 
        codes=codes, 
        dim=dim
    )
    vis_interval = quantizer.update_interval * 2

    datas1 = torch.randn(batchsize, heads, tokens, dim).abs()
    datas2 = torch.randn(batchsize, heads, tokens, dim).abs()*-1
    datas = torch.concat([datas1, datas2], dim=0)
    datas = datas / torch.norm(datas, dim=-1, keepdim=True)             # 将数据缩放至单位球

    with torch.no_grad():
        pass
        # for epoch in tqdm(torch.arange(1, epochs+1)):
        for epoch in torch.arange(1, epochs+1):
            delta_onehot, c = quantizer(datas)                          # [B, heads, tokens, S], [heads, S, dim]

            if epoch % vis_interval == 0:
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

            datas_hat = torch.einsum('bhls,hsk->bhlk', delta_onehot, c)
            error_quantization = torch.norm(datas.detach() - datas_hat.detach(), dim=-1).square().mean()
            print('余弦相似度量化损失:', error_quantization)

        error_quantization = torch.norm(datas.detach() - datas.detach().mean(dim=-2), dim=-1).square().mean()
        print('均值量化损失：', error_quantization)

        # # --- 可视化自注意力 ---
        # attn = torch.softmax(torch.einsum('bhld,bhnd->bhln', datas, datas), dim=-1).view(tokens, tokens)
        # attn_hat = torch.softmax(torch.einsum('bhld,bhnd->bhln', datas, datas_hat), dim=-1).view(tokens, tokens)
        # # # 可视化注意力图
        # GradVis(attn.cpu().detach(), 'attn', path_root + 'attention_map.png')
        # GradVis(attn_hat.cpu().detach(), 'attn_hat', path_root + 'attention_map_quantization.png')
        # # 可视化单个token的注意力
        # index_token = 0
        # x = torch.arange(1, attn.shape[-1]+1)
        # path_hist = path_root + 'softmax_of_token_{:04d}.png'.format(index_token)
        # plt.bar(x, attn[index_token].cpu().numpy(), color='skyblue')  
        # plt.title('References Distribution')
        # plt.xlabel('code index')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.savefig(path_hist, dpi=300)
        # plt.clf()
        # plt.close()
        # path_hist = path_root + 'softmax_of_quantization_token_{:04d}.png'.format(index_token)
        # plt.bar(x, attn_hat[index_token].cpu().numpy(), color='skyblue')  
        # plt.title('References Distribution')
        # plt.xlabel('code index')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.savefig(path_hist, dpi=300)
        # plt.clf()
        # plt.close()

        # # print(quantizer.c.data)
        # print('码表中更新次数最多的向量的均值：', quantizer.c[:, quantizer.c_count.argmax(dim=-1)].mean())
        # print('数据的均值：', datas.mean())

        # 可视化码表向量更新次数
        num_update = quantizer.c_count.squeeze().detach().cpu().sort()[0]
        x = np.arange(num_update.shape[0])
        path_hist = path_root + 'num_update.png'
        plt.bar(x, num_update.cpu().numpy(), color='skyblue')  
        plt.title('update times of vectors in codebook')
        plt.xlabel('code index')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(path_hist, dpi=300)
        plt.close()
