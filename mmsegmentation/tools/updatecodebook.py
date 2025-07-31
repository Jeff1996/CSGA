# 将预训练模型再目标数据集上推理过程中构造的码表，插入预训练权重
import torch

if __name__ == "__main__":
    path_src_net = '/home/student/hjf/workspace/mmsegmentation/work_dirs/swin-tiny-patch4-window7-LN_upernet_2xb1-80k_ade20k-512x512/iter_80000_QKL2norm_scale30.pth'
    path_src_codebooks = '/home/student/hjf/workspace/mmsegmentation/work_dirs/swin-tiny-patch4-window7-LN_upernet_2xb1-80k_ade20k-512x512/codebooks.pth'
    path_dst = '/home/student/hjf/workspace/mmsegmentation/work_dirs/swin-tiny-patch4-window7-LN_upernet_2xb1-80k_ade20k-512x512/upernet_swin_tiny_patch4_window7_512x512.pth'

    params_net = torch.load(path_src_net)
    params_codebooks = torch.load(path_src_codebooks)

    for key in params_codebooks:
        params_net['state_dict']['backbone.'+key] = params_codebooks[key]

    torch.save(params_net, path_dst)
    print('Done!')
    