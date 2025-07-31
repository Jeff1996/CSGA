_base_ = [
    '../_base_/models/fpn_pvt.py',
    '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/cityscapes'
crop_size = (768, 768)
ratio_subsample = 4         # 输入图片初步特征嵌入的分辨率下降比例
# cluster_size = (16, 16)
cluster_size = (24, 24)
stride = tuple(int(crop_size[i] / ratio_subsample / cluster_size[i]) for i in range(cluster_size.__len__()))

train_dataloader = dict(
    batch_size=4, 
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            img_path='image/train', seg_map_path='label/train'
        ),
    )
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            img_path='image/val', seg_map_path='label/val'
        ),
    )
)

test_dataloader = val_dataloader

# 模型配置
data_preprocessor = dict(
    size=crop_size
)

checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/05pvt_debug/epoch_50.pth'
# checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/05pvt_debug/epoch_50_img_size_768x768.pth'

model = dict(
    # type='EncoderDecoderMod',
    type='EncoderDecoder',
    # pretrained='pretrained/pvt_tiny.pth',
    # pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),

    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='pvt_tinyMod',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),
        # img_size=768,
        patch_size=4,
        embed_dims=[64, 128, 256, 512], 
        num_heads=[1, 2, 4, 8],             # 需要改成1248，以匹配聚类稀疏全局注意力对头数的要求
        out_indices=(0, 1, 2, 3),
        qk_scale=15.0,
        stride_cluster=stride,      # 构造初始聚类中心的步长
        with_cp=False,
    ),
    neck=dict(
        in_channels=[64, 128, 256, 512]
    ),
    decode_head=dict(
        num_classes=19
    )
)

# 学习策略配置
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer = dict(
        type='AdamW', 
        lr=0.0001, 
        weight_decay=0.0001
    )
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False
    )
]

train_cfg = dict(
    max_iters=40000,
    val_interval=4000
)

