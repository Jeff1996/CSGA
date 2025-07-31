_base_ = [
    '../_base_/models/twins_svt_base.py',
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

# model settings
checkpoint = '/home/hjf/workspace/mmpretrain/work_dirs/pretrained_in1k/07twins/twins-svt-small_3rdparty_8xb128_in1k_20220126-8fe5205b.pth'

model = dict(
    backbone=dict(
        arch='small'
    ), 
    head=dict(
        in_channels=512
    ),
    init_cfg=[
        dict(type='Pretrained', checkpoint=checkpoint)
    ],
)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4 * num_gpus * batch_size_pergpu * accumulative_counts / 512,
        # lr=5e-4 * 128 * 8 / 512,  # learning rate for 128 batch size, 8 gpu.
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        _delete=True, 
        norm_decay_mult=0.0, 
        bias_decay_mult=0.0
    ),
    clip_grad=dict(max_norm=5.0),
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
