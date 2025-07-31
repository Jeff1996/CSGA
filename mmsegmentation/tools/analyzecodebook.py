# codebook研究
import numpy as np
import matplotlib.pyplot as plt

import torch
import time

# 日志输出函数
def LogOut(path, text, mode):
    with open(path, mode, encoding='utf-8') as f:
        for line in text:
            line = '{}\n'.format(line)
            f.writelines(line)

# 可视化网络参数分布情况
def ParamsDisVis(path_save: str, weights: np.ndarray):
    # 绘制权重分布直方图
    plt.hist(weights, bins=100, density=False, alpha=0.6, color='g')
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(path_save, dpi=300)
    plt.close()

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

if __name__ == "__main__":
    path_root = '/home/student/hjf/workspace/mmsegmentation/work_dirs/swin-tiny-patch4-window7-LN_upernet_2xb1-80k_ade20k-512x512/'

    path_codebooks = path_root + 'iter_160000.pth'

    path_save = path_root + 'codebooks.txt'
    LogOut(path_save, ['码表分布情况统计-{}\n'.format(time.strftime("%Y-%m-%d_%H-%M-%S"))], 'w')
    codebooks_dict = torch.load(path_codebooks)['state_dict']
    for key in codebooks_dict.keys():
        if 'quantizer.c' in key and not 'quantizer.c_' in key:
            codebooks = codebooks_dict[key]
            # LogOut(path_save, ['------ {} ------'.format(key)], 'a')
            for index_head, codebook in enumerate(codebooks):
                print(codebook[:2])
                # 码表自注意力可视化(单位点积/余弦相似度)
                path_save_attn = path_root + 'self-attention_{}_head_{}.png'.format(key, index_head)
                print(codebook.shape)
                attn = torch.einsum('lc,mc->lm', codebook*30, codebook)
                attn = torch.softmax(attn, dim=-1)
                GradVis(attn.cpu(), 'head{}'.format(index_head), path_save_attn)
                continue

                LogOut(path_save, ['--- head: {} ---'.format(index_head)], 'a')

                # 整体信息统计
                codebook_max = torch.max(codebook)
                LogOut(path_save, ['最大值: ', codebook_max], 'a')

                codebook_min = torch.min(codebook)
                LogOut(path_save, ['最小值: ', codebook_min], 'a')

                codebook_mean = torch.mean(codebook)
                LogOut(path_save, ['均值: ', codebook_mean], 'a')

                codebook_std = torch.std(codebook)
                LogOut(path_save, ['方差: ', codebook_std], 'a')

                # 模信息统计(对于单位化的码表没有意义)
                codebook_norm = torch.norm(codebook, dim=-1)    # [codes, ]
                LogOut(path_save, ['模长: ', codebook_norm], 'a')

                codebook_norm_max = torch.max(codebook_norm)
                LogOut(path_save, ['模长-最大值: ', codebook_norm_max], 'a')

                codebook_norm_min = torch.min(codebook_norm)
                LogOut(path_save, ['模长-最小值: ', codebook_norm_min], 'a')

                codebook_norm_mean = torch.mean(codebook_norm)
                LogOut(path_save, ['模长-均值: ', codebook_norm_mean], 'a')

                codebook_norm_std = torch.std(codebook_norm)
                LogOut(path_save, ['模长-方差: ', codebook_norm_std], 'a')
                
                LogOut(path_save, ['\n'], 'a')

                # 对码表模长的直方统计
                path_hist = path_root + 'codebook_{}_head_{}.png'.format(key, index_head)
                ParamsDisVis(path_hist, codebook_norm.numpy())

                break
        elif 'quantizer.c_count' in key and not 'quantizer.c_count_' in key:
            codebooks = codebooks_dict[key]
            LogOut(path_save, ['------ {} ------'.format(key)], 'a')
            for index_head, codebook in enumerate(codebooks):
                LogOut(path_save, ['--- head: {} ---'.format(index_head)], 'a')
                x = np.arange(codebook.shape[0])
                codebook = codebook.sort()[0]
                LogOut(path_save, ['码表向量更新次数: ', codebook], 'a')

                # 可视化
                path_hist = path_root + 'codebook_{}_head_{}.png'.format(key, index_head)
                plt.bar(x, codebook.cpu().numpy(), color='skyblue')  
                plt.title('References Distribution')
                plt.xlabel('code index')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.savefig(path_hist, dpi=300)
                plt.close()

                # break
