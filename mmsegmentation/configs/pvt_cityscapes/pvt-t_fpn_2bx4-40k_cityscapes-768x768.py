_base_ = [
    '../_base_/models/fpn_pvt.py',
    '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/cityscapes'
crop_size = (768, 768)


train_dataloader = dict(
    batch_size=4, 
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            img_path='image/train', seg_map_path='label/train'
        ),
    )
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            img_path='image/val', seg_map_path='label/val'
        ),
    )
)

test_dataloader = val_dataloader


# 模型配置
data_preprocessor = dict(
    size=crop_size
)

checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/05pvt/epoch_50.pth'
# checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/05pvt/epoch_50_img_size_768x768.pth'

model = dict(
    # type='EncoderDecoderMod',
    type='EncoderDecoder',
    # pretrained='pretrained/pvt_tiny.pth',
    # pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),

    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='pvt_tiny',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),  # 模型编写时没有使用mmsegmentation的基类，所以这样加载权重是无效的
        # img_size=768,                     # 这里本来应该是模型训练时的标准图片尺寸，会影响位置编码的构造，但是由于pvt中没有处理预训练模型与当前模型的位置编码长度差异，所以暂时就用ImageNet-1K预训练时的图片尺寸224
        patch_size=4,
        embed_dims=[64, 128, 256, 512], 
        num_heads=[1, 2, 4, 8],             # 需要改成1248，以匹配聚类稀疏全局注意力对头数的要求
        out_indices=(0, 1, 2, 3),
        qk_scale=None,
        with_cp=False,
    ),
    neck=dict(
        in_channels=[64, 128, 256, 512]
    ),
    decode_head=dict(
        num_classes=19
    )
)

# 学习策略配置
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer = dict(
        type='AdamW', 
        lr=0.0001,
        weight_decay=0.0001
    )
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False
    )
]

train_cfg = dict(
    max_iters=40000,
    val_interval=4000
)

