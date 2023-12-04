from mmengine.config import Config

"""Configファイルが何をしているか確認してみよう"""
cfg = Config.fromfile("./configs/yolox/yolox_l_8xb8-300e_coco.py")
print(cfg)
cfg.dump("sample_dump.py")
