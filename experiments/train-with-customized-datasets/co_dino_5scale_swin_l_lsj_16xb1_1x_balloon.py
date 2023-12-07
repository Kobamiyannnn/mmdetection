_base_ = "../../projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_1x_coco.py"

num_classes = 1
data_root = "data/balloon/"  # defined in common/ssj_scp_270k_coco-instance.py
metainfo = {
    "classes": ("balloon",),
    "palette": [
        (220, 20, 60),
    ],
}
train_dataloader = dict(
    batch_size=1,  # defined in co_dino_5scale_swin_l_lsj_16xb1_1x_coco.py
    dataset=dict(
        dataset=dict(
            data_root=data_root,  # defined in common/ssj_scp_270k_coco-instance.py
            metainfo=metainfo,
            # defined in common/ssj_scp_270k_coco-instance.py
            ann_file="train/annotation_coco.json",
            # defined in common/ssj_scp_270k_coco-instance.py
            data_prefix=dict(img="train/")
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,  # defined in common/ssj_270k_coco-instance.py
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val/annotation_coco.json",
        data_prefix=dict(img="val/"),
    ),
)
test_dataloader = val_dataloader

model = dict(
    backbone=dict(
        # with_cp: defined in co_dino_5scale_swin_l_lsj_16xb1_1x_coco.py
        with_cp=True  # whether checkpoints are used or not.
    ),
    query_head=dict(
        transformer=dict(
            encoder=dict(
                # num_layers=6 defined in co_dino_5scale_r50_lsj_8xb2_1x_coco.py
                with_cp=6  # number of layers that use checkpoint.
            )
        )
    )
)

# defined in common/ssj_270k_coco-instance.py
val_evaluator = dict(
    # defined in common/ssj_270k_coco-instance.py
    ann_file=data_root + "val/annotation_coco.json",
    metric="bbox",  # defined in co_dino_5scale_r50_lsj_8xb2_1x_coco.py
)
test_evaluator = val_evaluator

# use the pre-trained CO-DETR model
# defined in _base_/default_runtime.py
load_from = "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth"  # noqa

# Hook config
default_hooks = dict(
    # defined in co_dino_5scale_r50_lsj_8xb2_1x_coco.py
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        save_best="coco/bbox_mAP",  # the best bbox_mAP of each val steps
    ),
)
env_cfg = dict(
    cudnn_benchmark=True,
)
# defined in co_dino_5scale_r50_lsj_8xb2_1x_coco.py
log_processor = dict(
    by_epoch=True
)

# auto_scale_lr is probably enable by default.
# defined in co_dino_5scale_r50_lsj_8xb2_1x_coco.py
auto_scale_lr = dict(enable=False, base_batch_size=16)
