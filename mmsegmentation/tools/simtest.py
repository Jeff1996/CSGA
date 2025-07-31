# 相似性度量方法比较
import os
import random
import numpy as np

import torch

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

# 计算余弦相似度
def getCS(a: torch.Tensor, b: torch.Tensor):
    '''
    a: [L, d]
    b: [S, d]
    '''
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    cs = torch.einsum('ld,sd->ls', a, b)
    return cs

if __name__ == '__main__':
    set_random_seed(0)

    L = 5
    S = L
    d = 1000
    a = torch.randn(L, d)
    b = torch.randn(S, d)

    ed = getED(a, b)
    cs = getCS(a, b)

    print('随机向量a: {}, b: {}的相似性度量: \n'.format(a.shape, b.shape))
    print('欧氏距离: \n{}'.format(ed))
    print('欧式距离方差: {}\n'.format(torch.var(ed)))
    print('欧式距离相对方差: {}\n'.format(torch.var(ed)/torch.abs(ed).mean()))

    print('余弦相似度: \n{}'.format(cs))
    print('余弦相似度方差: {}\n'.format(torch.var(cs)))
    print('余弦相似度相对方差: {}\n'.format(torch.var(cs)/torch.abs(cs).mean()))
