_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# 数据集配置（单卡）
data_root = '/mnt/ssd/hjf/ImageNet'
num_gpus = 2
batch_size_pergpu = 64

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
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', 
        pretrain_img_size=224,
        embed_dims=64,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 10, 4],
        num_heads=[2, 4, 8, 16],
        strides=(4, 2, 2, 2),
        # out_indices=(0, 1, 2, 3),
        out_indices=(3, ),
        qkv_bias=True,
        qk_scale=30.0,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        _delete_=True
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)


# 学习策略配置
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
    # optimizer=dict(
    #     type='AdamW',
    #     lr=5e-4 * num_gpus * batch_size_pergpu / 512,
    #     weight_decay=0.05,
    #     eps=1e-8,
    #     betas=(0.9, 0.999)
    # ),
    # paramwise_cfg=dict(
    #     norm_decay_mult=0.0,
    #     bias_decay_mult=0.0,
    #     flat_decay_mult=0.0,
    #     custom_keys={
    #         '.absolute_pos_embed': dict(decay_mult=0.0),
    #         '.relative_position_bias_table': dict(decay_mult=0.0)
    #     }
    # ),
    clip_grad=dict(max_norm=5.0)
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True
    ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR', 
        eta_min=1e-5, 
        by_epoch=True, 
        begin=20
    )
]

# train, val, test setting
train_cfg = dict(
    by_epoch=True, 
    max_epochs=300, 
    val_interval=10
)

default_hooks = dict(
    # 每 100 次迭代打印一次日志。
    logger=dict(type='LoggerHook', interval=100),
    # 每隔interval个epoch保存一次权重
    checkpoint=dict(type='CheckpointHook', interval=10)
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=num_gpus * batch_size_pergpu)
