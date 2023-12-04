import mmcv
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

# model init
config_file = "../checkpoints/rtmdet_tiny_8xb32-300e_coco.py"
checkpoint_file = "../checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
model = init_detector(config_file, checkpoint_file, device="cuda:0")

# inference
img = mmcv.imread("../demo/demo.jpg")
result = inference_detector(model, img)

# visualize result
img = mmcv.imconvert(img, "bgr", "rgb")
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
visualizer.add_datasample(
    "result",
    img,
    data_sample=result,
    draw_gt=False,
    show=False,
    out_file="../output/output.jpg",
)
