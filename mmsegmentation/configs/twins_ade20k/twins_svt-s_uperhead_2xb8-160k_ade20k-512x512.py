_base_ = [
    '../_base_/models/twins_pcpvt-s_upernet.py',
    '../_base_/datasets/ade20k.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/ade20k'
crop_size = (512, 512)


train_dataloader = dict(
    batch_size=8,
    num_workers=4, 
    dataset=dict(
        data_root=data_root,
    )
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    dataset=dict(
        data_root=data_root,
    )
)

test_dataloader = val_dataloader


# 模型配置
data_preprocessor = dict(
    size=crop_size
)

checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/07twins/epoch_50_seg.pth'

model = dict(
    type='EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SVT',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 4],
        windiow_sizes=[7, 7, 7, 7],
        norm_after_stage=True
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=150,
    )
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00006,             # batch size = 16
        betas=(0.9, 0.999), 
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
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
        power=1.0,
        begin=1500,
        end=160000,
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