_base_ = [
    '../_base_/models/upernet_swin.py', 
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/ade20k'
crop_size = (512, 512)

train_dataloader = dict(
    batch_size=8, 
    dataset=dict(
        data_root=data_root,
    )
)

val_dataloader = dict(
    batch_size=1, 
    dataset=dict(
        data_root=data_root,
    )
)

test_dataloader = val_dataloader


# 模型配置
# backbone_norm_cfg = dict(type='IN1d', requires_grad=True)
data_preprocessor = dict(
    size=crop_size
)

checkpoint_file = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/04swin/epoch_50.pth'

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.'),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,   # 默认是0.3
        patch_norm=True, 
    ),
    decode_head=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        in_channels=[96, 192, 384, 768], 
        num_classes=150,
    ),
    auxiliary_head=dict(in_channels=384, num_classes=150)
)

# 学习策略配置
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00006, 
        # lr=0.00003, 
        betas=(0.9, 0.999), 
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=1e-6, 
        by_epoch=False, 
        begin=0, 
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,          # 默认值
        begin=1500,
        end=160000,         # 默认值
        by_epoch=False,
    )
]

train_cfg = dict(
    max_iters=160000,       # 默认值
    # max_iters=80000,
    val_interval=16000
)

# 开启推理过程可视化
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)