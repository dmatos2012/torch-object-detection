import unittest
from unittest.mock import patch

import numpy as np
from dataset.transforms import RandomResizePad, ResizePad
from numpy.testing import assert_array_equal
from PIL import Image


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.ann = dict(
            boxes=np.array(([[112, 112, 336, 336]]), dtype=np.float64),
            labels=np.array(([1])),
        )
        self.input_img = np.random.random((448, 448))
        self.pil_img = Image.fromarray(self.input_img)
        self.target_size = 224

    @patch("random.uniform")
    def test_randomresizepad(self, mock_scale_factor):
        mock_scale_factor.return_value = 1.0
        interpolation = "random"
        fill_color = (0, 0, 0)
        random_interp = (Image.BILINEAR, Image.BICUBIC)
        img_tf = RandomResizePad(
            self.target_size, interpolation=interpolation, fill_color=fill_color
        )
        new_img, new_ann = img_tf(self.pil_img, self.ann)
        assert new_img.size[0] == 224
        assert new_img.size[1] == 224
        msg = "not expected interpolation"
        assert_array_equal(img_tf.interpolation, random_interp, msg)

        # Test different interpolation
        interpolation = "lanczos"
        img_tf = RandomResizePad(
            self.target_size, interpolation=interpolation, fill_color=fill_color
        )
        assert img_tf.interpolation == Image.LANCZOS

    def test_resizepad(self):
        interpolation = "bicubic"
        fill_color = (0, 0, 0)
        img_tf = ResizePad(
            self.target_size, interpolation=interpolation, fill_color=fill_color
        )
        new_img, new_ann = img_tf(self.pil_img, self.ann)
        assert new_img.size[0] == 224
        assert new_img.size[1] == 224
        assert img_tf.interpolation == "bicubic"
