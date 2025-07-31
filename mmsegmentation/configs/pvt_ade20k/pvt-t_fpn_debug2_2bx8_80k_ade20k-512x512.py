_base_ = [
    '../_base_/models/fpn_pvt.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
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
data_preprocessor = dict(
    size=crop_size
)

checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/05pvt_debug2/epoch_50.pth'

model = dict(
    type='EncoderDecoder',
    # pretrained='pretrained/pvt_tiny.pth',
    # pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),

    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='pvt_tinyMod2',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),
        # img_size=512,
        patch_size=4,
        embed_dims=[64, 128, 256, 512], 
        num_heads=[1, 2, 4, 8],             # 需要改成1248，以匹配聚类稀疏全局注意力对头数的要求
        out_indices=(0, 1, 2, 3),
        qk_scale=15.0,
        with_cp=False,
    ),
    neck=dict(
        in_channels=[64, 128, 256, 512]
    ),
    decode_head=dict(
        num_classes=150
    )
)

# 学习策略配置
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer = dict(
        type='AdamW', 
        lr=0.0001,              # batch size = 16
        weight_decay=0.0001
    )
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False
    )
]

train_cfg = dict(
    max_iters=80000,
    val_interval=8000
)
