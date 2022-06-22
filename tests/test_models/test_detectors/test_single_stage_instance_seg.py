# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet.core import DetDataSample
from mmdet.testing import demo_mm_inputs, get_detector_cfg
from mmdet.utils import register_all_modules

register_all_modules()


class TestSingleStageInstanceSegmentor(TestCase):

    @parameterized.expand([
        'solo/solo_r50_fpn_1x_coco.py',
        'solo/decoupled_solo_r50_fpn_1x_coco.py',
        'solo/decoupled_solo_light_r50_fpn_3x_coco.py',
        'solov2/solov2_r50_fpn_1x_coco.py',
        'solov2/solov2_light_r18_fpn_3x_coco.py'
    ])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        detector = build_detector(model)
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.mask_head)

    @parameterized.expand([
        ('solo/solo_r50_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('solo/decoupled_solo_r50_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('solo/decoupled_solo_light_r50_fpn_3x_coco.py', ('cpu', 'cuda')),
        ('solov2/solov2_r50_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('solov2/solov2_light_r18_fpn_3x_coco.py', ('cpu', 'cuda')),
    ])
    def test_single_stage_forward_loss_mode(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = build_detector(model)
            detector.init_weights()

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(
                2, [[3, 128, 128], [3, 125, 130]], with_mask=True)
            batch_inputs, data_samples = detector.data_preprocessor(
                packed_inputs, True)
            losses = detector.forward(batch_inputs, data_samples, mode='loss')
            self.assertIsInstance(losses, dict)

    @parameterized.expand([
        ('solo/solo_r50_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('solo/decoupled_solo_r50_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('solo/decoupled_solo_light_r50_fpn_3x_coco.py', ('cpu', 'cuda')),
        ('solov2/solov2_r50_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('solov2/solov2_light_r18_fpn_3x_coco.py', ('cpu', 'cuda')),
    ])
    def test_single_stage_forward_predict_mode(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = build_detector(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(
                2, [[3, 128, 128], [3, 125, 130]], with_mask=True)
            batch_inputs, data_samples = detector.data_preprocessor(
                packed_inputs, False)
            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(
                    batch_inputs, data_samples, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], DetDataSample)
