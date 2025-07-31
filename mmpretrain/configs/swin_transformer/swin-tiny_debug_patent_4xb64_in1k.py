_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# 数据集配置
data_root = '/mnt/ssd/hjf/ImageNet'
num_gpus = 8
batch_size_pergpu = 24
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

# 骨干网络（backbone前缀）+分类头权重（head前缀）
# checkpoint_file = '/home/hjf/workspace/mmpretrain/work_dirs/swin-tiny_debug_2xb64_in1k/swin_tiny_patch4_window7_224_modkey.pth'

# dict_keys(['meta', 'state_dict'])，params['state_dict']为网络权重（骨干网络（backbone前缀）+分类头权重（head前缀））
checkpoint_file = '/home/hjf/workspace/mmpretrain/work_dirs/pretrained_in1k/04swin/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'

# # 训练50epoch的结果
# checkpoint_file = '/home/hjf/workspace/mmpretrain/work_dirs/swin-tiny_debug_4xb64_in1k/epoch_50.pth'

# 只有骨干网络的权重，不含“backbone”前缀
# checkpoint_file = '/home/hjf/workspace/mmpretrain/work_dirs/swin-tiny_debug_2xb64_in1k/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'

# dict_keys(['model'])，只有骨干网络的权重，不含“backbone”前缀，并且各参数名称与mmlab中的不一致，无法直接使用
# checkpoint_file = '/home/hjf/workspace/mmpretrain/work_dirs/swin-tiny_debug_2xb64_in1k/swin_tiny_patch4_window7_224.pth'

# # 只更新量化模块权重
# checkpoint_file = '/home/hjf/workspace/mmpretrain/work_dirs/swin-tiny_debug_4xb64_in1k/epoch_50_聚类_无量化损失_只更新量化模块参数_80.642.pth'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformerPatent', 
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.'),
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        # out_indices=(0, 1, 2, 3),
        out_indices=(3, ),
        qkv_bias=True,
        # qk_scale=None,
        qk_scale=15.0,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        with_cp=True,               # 开启梯度检查点以节省内存，仅降低训练速度
        _delete_=True
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='head.'),
        num_classes=1000,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False
    ),
    # init_cfg=[
    #     dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
    #     dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    # ],
    init_cfg=[
        dict(type='Pretrained', checkpoint=checkpoint_file)
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
    optimizer=dict(
        type='AdamW',
        # filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4 * num_gpus * batch_size_pergpu * accumulative_counts / 512,
        # lr=1e-3 * num_gpus * batch_size_pergpu * accumulative_counts / 1024,       # 从零开始训练
        # lr = 4e-4 * num_gpus * batch_size_pergpu * accumulative_counts / 1024,    # 使用swin-t的预训练权重
        # lr = 8e-5 * num_gpus * batch_size_pergpu * accumulative_counts / 1024,    # 使用迭代50次的权重
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'patch_embed': dict(lr_mult=0, decay_mult=0),
    #         'downsample': dict(lr_mult=0, decay_mult=0),
    #         'blocks.0': dict(lr_mult=0, decay_mult=0),
    #         'blocks.2': dict(lr_mult=0, decay_mult=0),
    #         'blocks.4': dict(lr_mult=0, decay_mult=0),
    #         'head': dict(lr_mult=0, decay_mult=0),
    #     }
    # ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0), 
            # 'scale': dict(decay_mult=0.0)
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
    checkpoint=dict(type='CheckpointHook', interval=5
    )
)
