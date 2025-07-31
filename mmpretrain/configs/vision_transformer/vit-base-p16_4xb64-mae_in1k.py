_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
# 数据集配置
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
checkpoint_file = '/home/hjf/workspace/mmpretrain/work_dirs/pretrained_in1k/01vit/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.'),   # 新增
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='head.'),       # 新增
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ])
)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        # lr=1e-4 * 4096 / 256,   # 对应batch_size = 4096的学习率
        lr=1e-4 * num_gpus * batch_size_pergpu / 256,   # 对应batch_size = num_gpus * batch_size_pergpu的学习率
        weight_decay=0.3,
        eps=1e-8,
        betas=(0.9, 0.95)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
    
)

# runtime settings
custom_hooks = [dict(type='EMAHook', momentum=1e-4)]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
# auto_scale_lr = dict(base_batch_size=num_gpus*batch_size_pergpu)    # 这是不正确的做法，但是已经这样了，只能后期再重新训练了
auto_scale_lr = dict(base_batch_size=4096)        # 正确做法应该是保留这个（保持官方值），并且在训练时指定--auto-scale-lr参数

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

