_base_ = [
    '../_base_/models/segformer_mit-b0.py', 
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# 数据集配置
data_root = '/mnt/ssd/hjf/ImageNet'
num_gpus = 4
batch_size_pergpu = 128
accumulative_counts = 1

train_dataloader = dict(
    batch_size=batch_size_pergpu, 
    dataset=dict(
        data_root=data_root,
        ann_file='train.txt',
        split='',
    )
)

val_dataloader = dict(
    batch_size=batch_size_pergpu, 
    dataset=dict(
        data_root=data_root,
        ann_file='test.txt',
        split='',
    )
)

test_dataloader = val_dataloader


# 模型配置
# checkpoint = '/home/hjf/workspace/mmpretrain/work_dirs/pretrained_in1k/03segformer/mit_b0_mmlab_debug.pth'
checkpoint = '/home/hjf/workspace/mmpretrain/work_dirs/pretrained_in1k/03segformer/mit_b0_mmlab_debug_heads_1248.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MixVisionTransformerMod2',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],         # 第三个block的heads从5->4，为了匹配聚类稀疏注意力中的聚类集合变化
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(3,),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=15.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cp=False,                  # 开启梯度检查点以节省内存，仅降低训练速度
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=256,                # embed_dims * num_heads[-1]
        init_cfg=None,                  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', 
            label_smooth_val=0.1, 
            mode='original'
        ),
        cal_acc=False
    ),
    init_cfg=[
        dict(type='Pretrained', checkpoint=checkpoint)
    ],
)

# 学习策略配置
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        # filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3 * num_gpus * batch_size_pergpu * accumulative_counts / 1024,    # 根据实际batch_size调整学习率
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=5.0),
    accumulative_counts=accumulative_counts                 # 梯度累积，实现大batch_size
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=5,
        # update by iter
        convert_to_iter_based=True
    ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR', 
        eta_min=1e-5, 
        by_epoch=True, 
        begin=5
    )
]

# train, val, test setting
train_cfg = dict(
    by_epoch=True, 
    max_epochs=50, 
    val_interval=5
)

default_hooks = dict(
    # 每 100 次迭代打印一次日志。
    logger=dict(type='LoggerHook', interval=100),
    # 每隔interval个epoch保存一次权重
    checkpoint=dict(type='CheckpointHook', interval=5
    )
)
