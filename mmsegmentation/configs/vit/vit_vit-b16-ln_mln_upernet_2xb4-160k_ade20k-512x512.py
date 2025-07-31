_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/ade20k.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
# 数据集配置（单卡）
data_root = '../00_datasets/ade20k'
accumulative_counts = 1                  # 梯度累积，实现大batch_size，所以160k迭代次数等效80k
crop_size = (512, 512)

train_dataloader = dict(
    batch_size=4, 
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
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/01vit/epoch_50_seg.pth',
    backbone=dict(
        drop_path_rate=0.1, 
        final_norm=True,
        with_cp=False,
    ),
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))

# 学习策略配置
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        # lr=0.00006,         # 默认学习率，batch_size = 16
        lr=0.00003,         # 因为batch_size只能达到8
        betas=(0.9, 0.999), 
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

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

