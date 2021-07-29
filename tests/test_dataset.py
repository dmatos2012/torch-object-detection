from unittest import TestCase
from unittest.mock import patch

import numpy as np
from dataset.loader import create_loader
from dataset.open_images_dataset import OpenImagesDataset
from dataset.parser_open_images import OpenImagesParser
from dataset.transforms import MyCompose
from numpy.testing import assert_array_equal
from PIL import Image


class TestNetwork(TestCase):
    ann1 = [
        dict(
            category_id=160,
            boxes=np.array(([10.0, 20.0, 30.0, 40.0])),
            area=1520,
            iscrowd=None,
        ),
        dict(
            category_id=160,
            boxes=np.array(([40.0, 50.0, 60.0, 70.0])),
            area=4200,
            iscrowd=None,
        ),
    ]

    @patch("dataset.open_images_dataset.OpenImagesDataset")
    @patch.object(OpenImagesParser, "_load_annotations")
    @patch.object(OpenImagesParser, "get_ann_info", side_effect=ann1)
    @patch("PIL.Image.open")
    def setUp(self, mock_pil_image, mock_parse, m, mock_data):
        """I need OpenImagesParser patched, so that when load_annotations is called
        then it doesnt call COCO. Without that, then COCO is called and it asks for label.json"""
        # self.parser = OpenImagesParser(m)
        self.img1_info = {"height": 500, "width": 500, "file_name": "img1.jpg"}
        self.img2_info = {"height": 1000, "width": 1000, "file_name": "img2.jpg"}

        self.info = [self.img1_info, self.img2_info]
        mock_data.parser.img_infos.return_value = self.info
        mock_data.parser.img_ids.return_value = [0, 1]
        self.pil_img = Image.fromarray(np.random.random((448, 448, 3)), mode="RGB")
        mock_pil_image.return_value = self.pil_img
        # mock_pil_image.assert_called()

        self.dataset = OpenImagesDataset("path_to_data", "dataset")
        # TODO use create_dataset to test my dataset rather than OpenImagesDataset(dataset_factory.py)
        self.dataset.parser.img_infos = mock_data.parser.img_infos.return_value
        self.dataset.parser.img_ids = mock_data.parser.img_ids.return_value
        # print(self.dataset.__getitem__(0))

    #
    @patch.object(OpenImagesParser, "get_ann_info", side_effect=ann1)
    @patch("PIL.Image.open")
    def test_getitem(self, *args):
        img0, target0 = self.dataset.__getitem__(0)
        assert target0["img_size"] == (500, 500)
        assert_array_equal(target0["boxes"], self.ann1[0]["boxes"])

        img1, target1 = self.dataset.__getitem__(1)
        assert target1["img_size"] == (1000, 1000)
        assert_array_equal(target1["boxes"], self.ann1[1]["boxes"])

    def test_loader(self):
        assert self.dataset.transform is None
        create_loader(
            self.dataset,
            input_size=224,
            batch_size=2,
            interpolation="bilinear",
            is_training=True,
        )
        assert isinstance(self.dataset.transform, MyCompose)
