_base_ = [
    '../_base_/models/pvt.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# 数据集配置（单卡）
data_root = '/mnt/ssd/hjf/ImageNet'
num_gpus = 4
batch_size_pergpu = 64
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

# dict_keys(['meta', 'state_dict'])，params['state_dict']为网络权重（骨干网络（backbone前缀）+分类头权重（head前缀））
checkpoint = '/home/hjf/workspace/mmpretrain/work_dirs/pretrained_in1k/05pvt/pvt_tiny_mmpretrain_heads_1248.pth'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='pvt_tiny',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),
        embed_dims=[64, 128, 256, 512], 
        num_heads=[1, 2, 4, 8],             # 需要改成1248，以匹配聚类稀疏全局注意力对头数的要求
        out_indices=(3, ),
        qk_scale=None,
        with_cp=False,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='head.'),       # 新增
        num_classes=1000,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    # init_cfg=[
    #     dict(type='TruncNormal', layer='Linear', std=.02),
    #     dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    # ],
    init_cfg=[
        dict(type='Pretrained', checkpoint=checkpoint)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ])
)


# 学习策略配置
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4 * num_gpus * batch_size_pergpu * accumulative_counts / 512,
        # lr = 4e-4 * num_gpus * batch_size_pergpu * accumulative_counts / 1024,    # 使用swin-t的预训练权重
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
        }
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
    checkpoint=dict(type='CheckpointHook', interval=5)
)















# 官方配置参数
# cfg = dict(
#     model='pvt_small',
#     drop_path=0.1,
#     clip_grad=None,
#     output_dir='checkpoints/pvt_small',
# )