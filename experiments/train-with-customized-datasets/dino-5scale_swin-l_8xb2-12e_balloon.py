# The new config inherits a base config to highlight the necessary modification
_base_ = "mmdet::dino/dino-5scale_swin-l_8xb2-12e_coco.py"

# We also need to change the num_classes in head to match the dataset's annotation
# num_cpを設定することで、Activation checkpointingを適用することができる
# 上限はlayer数
# deformable_detr_layers.pyでcheckpoint_wrapper(self.layers[i], offload_to_cpu=True)
# とするとさらにメモリ使用量を下げられそう
#
# backboneに関してもwith_cpがTrueになっているため、Activation checkpointingされている
model = dict(bbox_head=dict(num_classes=1), encoder=dict(num_cp=6))

# Modify dataset related settings
data_root = "data/balloon/"
metainfo = {
    "classes": ("balloon",),
    "palette": [
        (220, 20, 60),
    ],
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train/annotation_coco.json",
        data_prefix=dict(img="train/"),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val/annotation_coco.json",
        data_prefix=dict(img="val/"),
    )
)
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + "val/annotation_coco.json")
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"  # noqa

auto_scale_lr = dict(base_batch_size=1)
# Learning rateの調整
"""
optim_wrapper = dict(
    optimizer=dict(
        lr=0.02 / 10
    )
)
"""

# Gradient Accumulation
"""
optimizer = dict(
    accumulative_counts=16
)
"""
