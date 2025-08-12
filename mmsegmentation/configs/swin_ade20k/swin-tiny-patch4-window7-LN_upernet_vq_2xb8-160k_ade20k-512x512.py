_base_ = [
    '../_base_/models/upernet_swin.py', 
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

# data setting
data_root = 'path/to/ade20k'
crop_size = (512, 512)

train_dataloader = dict(
    batch_size=8,                               # 8 (batch size) * 2 (GPUS) = 16
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

# model setting
data_preprocessor = dict(
    size=crop_size
)
checkpoint = 'path/to/ImageNet-1K/pre-trained/weight.pth'
model = dict(
    type='EncoderDecoderVQ',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformerVQ',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        qk_scale=None,
        codes=256,
        drop_path_rate=0.3,
        patch_norm=True, 
        with_cp=False,                          # gradient checkpoint
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768], 
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=384, 
        num_classes=150,
    )
)

# optimization setting
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00006,                             # bs = 16
        betas=(0.9, 0.999), 
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }
    ),
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
    max_iters=160000,
    val_interval=16000
)
