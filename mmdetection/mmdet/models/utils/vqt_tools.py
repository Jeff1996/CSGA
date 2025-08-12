# 工具包
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool

import torch
import torch.nn.functional as F
import torch.distributed as dist

# 导入二分聚类函数
try:
    # C++编写的二分聚类程序，源文件路径在utils/BinaryCluster.cpp，需要python setup.py install安装后使用
    import BinaryCluster
    Flag_binarycluster_cpp = True
except:
    print('!!!!!!没有安装C++优化的BinaryCluster, 故采用慢速python实现!!!!!!')
    Flag_binarycluster_cpp = False

# 导入量化聚类函数
try:
    # C++编写的二分聚类程序，源文件路径在utils/QuantizationCluster.cpp，需要python setup.py install安装后使用
    import QuantizationCluster
    Flag_quantizationcluster_cpp = True
except:
    print('!!!!!!没有安装C++优化的QuantizationCluster, 故采用慢速python实现!!!!!!')
    Flag_quantizationcluster_cpp = False

# 调试开关
Flag_debug = False

# # 屏蔽C++程序
# Flag_quantizationcluster_cpp = False
# Flag_binarycluster_cpp = False

# 计算多卡变量的求和
def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

# 日志输出函数
def LogOut(path, text, mode):
    with open(path, mode, encoding='utf-8') as f:
        for line in text:
            line = '{}\n'.format(line)
            f.writelines(line)

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

# 将tokens序列进行分块(先恢复tokens序列的二维形式，再以patch形式分块，从而保证相邻token被分到同一个block)
def getBlocks(x: torch.Tensor, stride: int=4):
    '''
    x       : [B, dim, H, W]
    stride  : 滑窗尺寸
    '''
    B, dim, H, W = x.shape
    R = (H // stride) * (W // stride)                       # number of blocks
    C = stride ** 2                                         # block size   
    x_unfold = F.unfold(                                    # [B, R, C, dim]
        x, 
        kernel_size=stride, 
        padding=0, 
        stride=stride
    ).view(B, dim, C, R).permute(0, 3, 2, 1).contiguous()   # [B, dim*C, R] -> [B, dim, C, R] -> [B, R, C, dim]
    return x_unfold

# 将分块的tokens还原为序列形式，与getBlocks方法的作用相反
def getFeaturemap(x_unfold: torch.Tensor, output_size: tuple, stride: int=4):
    '''
    x       : [B, R, C, dim]
    stride  : 滑窗尺寸
    '''
    B, R, C, dim = x_unfold.shape
    x_unfold = x_unfold.permute(0, 3, 2, 1).contiguous().view(B, -1, R) # [B, R, C, dim] -> [B, dim, C, R] -> [B, dim*C, R]
    x_2d = F.fold(                                                      # [B, dim, H, W]
        x_unfold,
        output_size=output_size,
        kernel_size=stride,
        stride=stride
    )
    return x_2d

# 相对位置编码
def getMask(stride: tuple=(4, 4), gain: float=1.0):
    assert type(stride) == tuple, 'Data type of stride in function <getMask> should be <tuple>!'
    assert stride.__len__() == 2, 'Length of stride should be 2!'
    coords_h = torch.arange(stride[0])
    coords_w = torch.arange(stride[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))                  # 构造坐标窗口元素坐标索引，[2, h, w]
    coords_flatten = torch.flatten(coords, start_dim = 1)                       # 一维化，[2, hw]
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]   # 构造相对位置矩阵, 第一个矩阵是h方向的相对位置差, 第二个矩阵是w方向的相对位置差
    distance = torch.sqrt(                                                      # [hw, hw]
        torch.square(relative_coords[0]) + torch.square(relative_coords[1])
    )
    distance_exp = torch.exp(distance)                                          # exp操作用于处理distance中的0, [hw, hw]
    mask = (1 / distance_exp) * gain                                            # 距离越远的token注意力增强越少(加性增强), 最大值为1*gain, 最小值可以接近0, [hw, hw]
    mask = mask.view(1, 1, *mask.shape)                                         # [1, 1, hw, hw]
    return mask

# 梯度可视化
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
    plt.close

# 保存网络配置信息
def SaveConfig(path_file: str, args: argparse.ArgumentParser):
    with open(path_file, 'w', encoding='utf-8') as f:
        line = '{:2s} {:20s} {:20s} {}\n'.format('序号', '参数名称', '参数类型', '参数值')
        f.writelines(line)
        for index, (param, value) in enumerate(args._get_kwargs()):
            line = '{:3d}, {:20s}: {:20s}, {}\n'.format(index, param, str(type(value)), value)
            f.writelines(line)

# 可视化网络参数分布情况
def ParamsDisVis(path_save: str, net: torch.nn.Module):
    weights = []
    for name, param in net.named_parameters():
        if 'weight' in name:
            weight = param.data.view(-1).cpu().numpy()
            weights.append(weight)  # 展平权重并转换为numpy数组

    # 合并权重（可选）
    if len(weights) > 1:
        weights = np.concatenate(weights)
    else:
        weights = weights[0]
    # print(weights.min(), weights.max())

    # 绘制权重分布直方图
    plt.hist(weights, bins=1000, density=True, alpha=0.6, color='g')
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(path_save, dpi=300)
    plt.close()

# Methods implemented in python
def BinaryCluster_py(datas: torch.tensor):
    '''
    datas: tensor, [B, L, dims]
    '''
    assert datas.dim() >= 2 and datas.dim() <= 3, 'dims of datas is out of range: 2-3.'
    if datas.dim() == 2:
        datas = datas.unsqueeze(0)
    B, L, dims = datas.shape

    # - 图片级循环 -
    deltas_list = []                                                # 量化索引列表: [Δ_1, Δ_2, ..., Δ_B]
    codebooks_list = []
    for data_index in range(B):
        tokens_cluster_index = [torch.arange(L)]                    # tokens索引, 随聚类过程一同更新, 用于构造deltas索引矩阵
        tokens_cluster = [datas[data_index]]                        # 聚类结果, 列表的每个元素都是一个聚类, 最开始只有一个聚类
        # - 通道级循环: 获取一张图片的codebook -
        for dim_index in range(dims):
            # - 聚类级循环 -
            tokens_cluster_index_temp = []
            tokens_cluster_temp = []
            for cluster_index in range(tokens_cluster_index.__len__()):
                tokens_index = tokens_cluster_index[cluster_index]  # 获取当前待分类的tokens的索引
                tokens = tokens_cluster[cluster_index]              # 获取当前待分类的tokens
                if tokens_index.shape[0] == 1:
                    tokens_cluster_index_temp.append(tokens_index)
                    tokens_cluster_temp.append(tokens)
                elif tokens_index.shape[0] > 1:
                    mean = tokens[:,dim_index].mean()
                    tokens_demean = tokens[:,dim_index] - mean
                    tokens_low_index = tokens_index[tokens_demean < 0]
                    tokens_low = tokens[tokens_demean < 0]
                    tokens_high_index =  tokens_index[tokens_demean >= 0]                   
                    tokens_high = tokens[tokens_demean >= 0]
                    if tokens_low_index.shape[0] > 0:
                        tokens_cluster_index_temp.append(tokens_low_index)
                        tokens_cluster_temp.append(tokens_low)
                    if tokens_high_index.shape[0] > 0:
                        tokens_cluster_index_temp.append(tokens_high_index)
                        tokens_cluster_temp.append(tokens_high)
                else:
                    pass
            tokens_cluster_index = tokens_cluster_index_temp
            tokens_cluster = tokens_cluster_temp
            if tokens_cluster_index.__len__() == L:                 # 如果当前的聚类数已经等于token数, 则停止后续分类
                break

        # - 计算聚类中心 -
        delta = torch.zeros(L)                                      # 保存量化索引: K ≈ K^ = ΔC
        tokens_center = []
        for cluster_index in range(tokens_cluster_index.__len__()):
            # delta
            tokens_index = tokens_cluster_index[cluster_index]      # 当前聚类所有成员token在K矩阵中的索引
            delta[tokens_index] = cluster_index                     # 分配量化索引
            # codebook
            tokens_singlecls = tokens_cluster[cluster_index]
            tokens_center.append(tokens_singlecls.mean(0))

        # - 码表补全 -
        paddings = 2 ** dims - tokens_center.__len__()
        if paddings > 0:
            token_padding = torch.zeros(dims)
            for _ in range(paddings):
                tokens_center.append(token_padding)                 # 如果码表长度不够则填充随机向量

        # - 保存当前图片的码表和量化索引 -
        deltas_list.append(delta)
        codebooks_list.append(torch.stack(tokens_center))

    deltas = torch.stack(deltas_list).long()
    codebooks = torch.stack(codebooks_list)
    return (deltas, codebooks)                                      # [B, L], [B, C, dim]

def binarycluster(datas: torch.tensor):
    if Flag_binarycluster_cpp:
        results = BinaryCluster.BinaryCluster_cpp(datas)
    else:
        results = BinaryCluster_py(datas)
    return results

# Methods implemented in python
def QuantizationCluster_py(datas: torch.tensor, levels: int=4):
    '''
    datas: tensor, [B, L, dims]
    levels: int, 量化等级
    '''
    assert datas.dim() >= 2 and datas.dim() <= 3, 'dims of datas is out of range: 2-3.'
    if datas.dim() == 2:
        datas = datas.unsqueeze(0)
    B, L, dims = datas.shape

    # - 量化 -
    datas_max = datas.max(dim=-2, keepdim=True)[0]                  # 
    datas_min = datas.min(dim=-2, keepdim=True)[0]                  # 
    datas_interval = (datas_max - datas_min) / levels * (1 + 1e-4)  # 
    datas_interval[datas_interval==0] = 1e-6                        # 防止除以0
    datas_quantized = torch.floor((datas - datas_min) / datas_interval).long()

    # - 图片级循环 -
    deltas_list = []                                                # 量化索引列表: [Δ_1, Δ_2, ..., Δ_B]
    codebooks_list = []                                             # 码表列表
    for data_index in range(B):
        tokens_cluster_index = [torch.arange(L)]                    # tokens索引, 随聚类过程一同更新, 用于构造deltas索引矩阵
        tokens_cluster = [datas_quantized[data_index]]              # 聚类结果, 列表的每个元素都是一个聚类, 最开始只有一个聚类
        # - 通道级循环: 获取一张图片的codebook -
        for dim_index in range(dims):
             # - 聚类级循环 -
            tokens_cluster_index_temp = []
            tokens_cluster_temp = []
            for cluster_index in range(tokens_cluster_index.__len__()):
                tokens_index = tokens_cluster_index[cluster_index]  # 获取当前待分类的tokens的索引
                tokens = tokens_cluster[cluster_index]              # 获取当前待分类的tokens
                if tokens_index.shape[0] == 1:                      # 当前类别只有一个token, 不再细分
                    tokens_cluster_index_temp.append(tokens_index)
                    tokens_cluster_temp.append(tokens)
                elif tokens_index.shape[0] > 1:                     # 当前类别不只一个token, 还需细分
                    classes_id = tokens[:, dim_index].unique()      # 获取当前特征维度的量化类别
                    for class_id in classes_id:                     # 根据当前维度的量化类别对tokens进行分类
                        tokens_cluster_index_temp.append(tokens_index[tokens[:,dim_index]==class_id])
                        tokens_cluster_temp.append(tokens[tokens[:,dim_index]==class_id])
                else:
                    pass
            tokens_cluster_index = tokens_cluster_index_temp
            tokens_cluster = tokens_cluster_temp
            if tokens_cluster_index.__len__() == L:                 # 如果当前的聚类数已经等于token数, 则停止后续分类
                break
        
        # - 计算聚类中心 -
        delta = torch.zeros(L)                                      # 保存量化索引: K ≈ K^ = ΔC
        tokens_center = []                                          # 保存聚类中心
        for cluster_index in range(tokens_cluster_index.__len__()):
            tokens_index = tokens_cluster_index[cluster_index]      # 当前聚类所有成员token在K矩阵中的索引
            # delta
            delta[tokens_index] = cluster_index                     # 分配量化索引
            # codebook
            tokens_singlecls = datas[data_index][tokens_index]      # 获取当前聚类所有成员token
            tokens_center.append(tokens_singlecls.mean(0))

        # - 码表补全 -
        paddings = levels ** dims - tokens_center.__len__()
        if paddings > 0:
            token_padding = torch.zeros(dims)
            for _ in range(paddings):
                tokens_center.append(token_padding)                 # 如果码表长度不够则填充随机向量

        if not tokens_center.__len__() == levels ** dims:
            LogOut('./tasks/classification/log_error.txt', ['码表长度与目标长度不匹配！\n'], 'a')
            LogOut('./tasks/classification/log_error.txt', ['codebooks: \n'], 'a')
            LogOut('./tasks/classification/log_error.txt', tokens_center, 'a')

        assert tokens_center.__len__() == levels ** dims, "码表长度与目标长度不匹配！"

        # - 保存当前图片的码表和量化索引 -
        deltas_list.append(delta)
        codebooks_list.append(torch.stack(tokens_center))

    deltas = torch.stack(deltas_list).long()
    codebooks = torch.stack(codebooks_list)
    return (deltas, codebooks)                                      # [B, L], [B, C, dim]

def quantizationcluster(datas: torch.tensor, levels: int=4):
    if Flag_quantizationcluster_cpp:
        results = QuantizationCluster.QuantizationCluster_cpp(datas, levels)
    else:
        results = QuantizationCluster_py(datas, levels)
    return results

# 聚类工具
class ClusterTool:
    def __init__(self, num_workers: int=2, method: str='binarycluster', levels: int=4) -> None:
        '''
        num_workers: int, 进程数, 默认为2
        method: str, 码表构造方法, 默认为binarycluster
        levels: int, 量化等级, 仅quantizationcluster方法会用到此参数
        '''
        self.num_workers = num_workers                              # 子进程数量，如果等于1，则不创建子进程（这由子进程控制方法进行处理）
        if num_workers > 1:
            # --- 创建子进程池 ---
            self.pool = Pool(num_workers)

        self.method_name = method
        # --- 确定基本的数据处理函数 ---
        if method == 'binarycluster':
            self.method_fun = binarycluster
        elif method == 'quantizationcluster':
            self.method_fun = quantizationcluster
        else:
            raise NotImplementedError

        self.levels = levels

    # 数据分发器
    def DataDistributor(self, datas: torch.tensor):
        '''
        datas: tensor, [B, L, dims]
        '''
        # 确定每个线程分配的数据量
        B, _, _ = datas.shape
        num_list = [B // self.num_workers] * self.num_workers       # 先保证每个进程都能分到差不多的任务
        for i in range(B % self.num_workers):                       # 剩下的一些任务就平分给前几个进程
            num_list[i] += 1

        # 分配数据
        datas_list = []
        index = 0
        for i, num in enumerate(num_list):
            if num == 0:                                            # 如果当前线程分不到数据，那后续线程也同样分不到
                break
            data = datas[index:index+num]
            if self.method_name == 'binarycluster':
                datas_list.append([data])
            elif self.method_name == 'quantizationcluster':
                datas_list.append([data, self.levels])
            else:
                raise NotImplementedError
            index += num
        return datas_list

    # 数据汇总
    def DataCollector(self, results):
        deltas_list = []
        codebooks_list = []
        for item in results:
            deltas_list.append(item[0])
            codebooks_list.append(item[1])
        deltas = torch.cat(deltas_list, dim=0)
        codebooks = torch.cat(codebooks_list, dim=0)
        return deltas, codebooks

    # 任务控制
    def Apply(self, datas: torch.tensor):
        '''
        datas: tensor, [B, L, dims]
        '''
        if self.num_workers <= 1:
            if self.method_name == 'binarycluster':
                deltas, codebooks = self.method_fun(datas)
            elif self.method_name == 'quantizationcluster':
                deltas, codebooks = self.method_fun(datas, self.levels)
            else:
                raise NotImplementedError
            return deltas, codebooks
        else:
            datas_list = self.DataDistributor(datas)                    # 给每个线程分配数据
            results = self.pool.starmap(self.method_fun, datas_list)    # 多线程处理, 输出结果以列表形式返回, 对应输入数据列表的顺序
            deltas, codebooks = self.DataCollector(results)             # 收集每个线程的处理结果
            return deltas, codebooks

# 分割指标统计工具
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = None                                     # 混淆矩阵（空）

    # 构造混淆矩阵
    def genConfusionMatrix(self, imgPredict: torch.Tensor, imgLabel: torch.Tensor, ignore_labels: list=[]):
        '''
        输入参数: 
        imgPredict      : tensor, [B, H, W] 或 [B, dim, H, W] 或 [H, W]
        imgLabel        : tensor, [B, H, W] 或 [B, dim, H, W] 或 [H, W]
        ignore_labels   : list, [n, ]

        输出参数: 
        confusionMatrix : tensor, [number_classes, number_classes], 单个batch数据构成的混淆矩阵
        '''
        if imgPredict.ndim > imgLabel.ndim:
            imgPredict = imgPredict.argmax(dim=-3, keepdim=False)       # [B, dim, H, W] -> [B, H, W]
        assert imgPredict.shape == imgLabel.shape

        mask = (imgLabel >= 0) & (imgLabel < self.numClass)             # 排除异常标签
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)                               # 排除指定类别
        label = self.numClass * imgLabel[mask] + imgPredict[mask]       # 非常有意思的混淆矩阵统计方法：numClass * imgLabel充当混淆矩阵中的纵坐标索引，imgPredict充当横坐标索引
        count = torch.bincount(label, minlength=self.numClass ** 2)     # torch.bincount方法统计混淆矩阵中各个位置元素出现的次数
        confusionMatrix = count.view(self.numClass, self.numClass)
        return confusionMatrix
    
    def addBatch(self, imgPredict: torch.Tensor, imgLabel: torch.Tensor, ignore_labels: list=[]):
        '''
        输入参数: 
        imgPredict      : tensor, [B, H, W] 或 [B, dim, H, W] 或 [H, W]
        imgLabel        : tensor, [B, H, W] 或 [B, dim, H, W] 或 [H, W]
        ignore_labels   : list, [n, ]

        输出参数: 
        confusionMatrix : tensor, [number_classes, number_classes], 所有batch数据的累计混淆矩阵
        '''
        # 累计混淆矩阵
        if not self.confusionMatrix is None:
            self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)
        else:
            self.confusionMatrix = self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)
        return self.confusionMatrix

    def reset(self):
        # self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
        self.confusionMatrix = None

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc                                     # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean()  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc                                      # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)     # 取对角元素的值，返回列表

        # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - intersection

        IoU = intersection / union                          # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()                 # 求各类别IoU的平均
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


