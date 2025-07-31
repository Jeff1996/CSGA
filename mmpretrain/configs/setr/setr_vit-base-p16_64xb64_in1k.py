_base_ = [
    '../_base_/models/vit-base-p16.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# 数据集配置
data_root = '/mnt/ssd/hjf/ImageNet'
num_gpus = 4
batch_size_pergpu = 64
accumulative_counts = 1

data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=224, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=batch_size_pergpu, 
    dataset=dict(
        data_root=data_root,
        ann_file='train.txt',
        split='',
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=batch_size_pergpu, 
    dataset=dict(
        data_root=data_root,
        ann_file='test.txt',
        split='',
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader


# model setting
checkpoint_file = '/home/hjf/workspace/mmpretrain/work_dirs/pretrained_in1k/01vit/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth'

model = dict(
    backbone=dict(
        type='VisionTransformer',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.'),   # 新增
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),   # 新增
        arch='b',
        img_size=224,
        with_cp=False,               # 开启梯度检查点以节省内存，仅降低训练速度
    )
)

# # runtime settings
# auto_scale_lr = dict(base_batch_size=num_gpus*batch_size_pergpu)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', 
        # lr=0.003,                                     # 对应batch_size = 4096的学习率
        lr=3e-3 * num_gpus * batch_size_pergpu / 4096,  # 对应batch_size = num_gpus * batch_size_pergpu的学习率
        weight_decay=0.3
    ),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
    clip_grad=dict(max_norm=1.0),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=5,
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