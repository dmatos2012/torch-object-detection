from unittest import TestCase
from unittest.mock import Mock, patch

import pytest
from torch_object_detection.dataset.open_images_dataset import OpenImagesDataset
from torch_object_detection.dataset.parser_open_images import OpenImagesParser
from typing_extensions import OrderedDict


def dummy_parse(ann_file):
    img_id = Mock()
    return img_id


# @patch('torch_object_detection.parser.parser_open_images.OpenImagesParser._load_annotations')
@pytest.mark.skip(reason="need to modify")
class TestNetwork(TestCase):
    def setUp(self):
        pass

    @patch(
        "torch_object_detection.parser.parser_open_images.OpenImagesParser._load_annotations"
    )
    def test_x(self, mock_parser):
        mock_parser.side_effect = dummy_parse
        OpenImagesParser("kahlua.json")._load_annotations("kahlua.json")
        mock_parser.assert_called()

    def test_len_dataset(self):
        path = "/home/david/fiftyone/open-images-v6"
        splits = ("train", "validation")
        dataset_cls = OpenImagesDataset
        datasets = OrderedDict()
        for s in splits:
            datasets[s] = dataset_cls(path, s)
        assert len(datasets["train"].parser.img_ids) == 1000
        assert datasets["train"].__len__() == 1000
        assert datasets["validation"].__len__() == 200

    def test_mock_works(self):
        annff = "/home/david/fiftyone/open-images-v6/train/labels.json"
        OpenImagesParser(annff)._load_annotations(annff)
        assert 1 == 1

    def test_input_image_size(self):
        pass
        # input_size = 224  # input of image
        # batch_size = 2
        # num_workers = 2
        # interpolation = "bilinear"
        # fill_color = "mean"
        # loader_train = create_loader(
        #     dataset_train,
        #     input_size,
        #     batch_size,
        #     interpolation=interpolation,
        #     fill_color=fill_color,
        #     num_workers=num_workers,
        #     is_training=True,
        # )

    def test_augmentations(self):
        pass

    def test_shape(self):
        pass
