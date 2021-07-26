from unittest import TestCase
from unittest.mock import patch

import pytest
from dataset.loader import create_loader
from dataset.open_images_dataset import OpenImagesDataset
from dataset.parser_open_images import OpenImagesParser
from dataset.transforms import MyCompose
from typing_extensions import OrderedDict


# @patch('torch_object_detection.parser.parser_open_images.OpenImagesParser._load_annotations')
# @pytest.mark.skip(reason="need to modify")
class TestNetwork(TestCase):
    def setUp(self):
        self.path = "/home/david/fiftyone/open-images-v6"
        self.splits = ("train", "validation")
        self.dataset_cls = OpenImagesDataset
        self.datasets = OrderedDict()
        for s in self.splits:
            self.datasets[s] = self.dataset_cls(self.path, s)
        self.train_dataset, self.val_dataset = list(self.datasets.values())

    @patch(
        "torch_object_detection.parser.parser_open_images.OpenImagesParser._load_annotations"
    )
    @pytest.mark.skip(reason="not ready yet")
    def test_x(self, mock_parser):
        # mock_parser.side_effect # a fcn
        OpenImagesParser("kahlua.json")._load_annotations("kahlua.json")
        mock_parser.assert_called()

    def test_len_dataset(self):
        assert self.train_dataset.__len__() == 1000
        assert self.val_dataset.__len__() == 200

    def test_loader(self):
        assert self.train_dataset.transform is None
        create_loader(
            self.train_dataset,
            input_size=224,
            batch_size=2,
            interpolation="bilinear",
            is_training=True,
        )
        assert isinstance(self.train_dataset.transform, MyCompose)
