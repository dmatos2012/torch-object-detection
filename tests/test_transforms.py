import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from torch_object_detection.dataset.transforms import RandomResizePad


class TestTransform(unittest.TestCase):
    @patch("random.uniform")
    def test_randomresizepad(self, mock_scale_factor):
        ann = dict(
            boxes=np.array(([[112, 112, 336, 336]]), dtype=np.float64),
            labels=np.array(([1])),
        )
        input_img = np.random.random((448, 448))
        pil_img = Image.fromarray(input_img)
        mock_scale_factor.return_value = 1.0
        target_size = 224
        interpolation = "random"
        fill_color = (0, 0, 0)
        img_tf = RandomResizePad(
            target_size, interpolation=interpolation, fill_color=fill_color
        )
        new_img, new_ann = img_tf(pil_img, ann)
        assert new_img.size[0] == 224
        assert new_img.size[1] == 224
