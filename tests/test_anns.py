from unittest import TestCase
from unittest.mock import patch

from torch_object_detection.dataset.parser_open_images import OpenImagesParser


# @patch('torch_object_detection.parser.parser_open_images.OpenImagesParser._load_annotations')
class TestNetwork(TestCase):
    # @patch('pycocotools.coco.COCO.getAnnIds')
    @patch("pycocotools.coco.COCO")
    @patch(
        "torch_object_detection.dataset.parser_open_images.OpenImagesParser._load_annotations"
    )
    def test_img_anno(self, load_anns, mock_coco):
        ann = [
            dict(
                id=10,
                image_id=1,
                category_id=160,
                bbox=[10, 20, 30, 40],
                area=1520,
                iscrowd=None,
            )
        ]
        # fix the category_id one, since now its forced to be either 160 or 96(my two classes)
        parser = OpenImagesParser("kahlua.json")
        parser.coco = mock_coco
        parser.coco.loadAnns.return_value = ann
        # parser._load_annotations("kahlua.json")
        img_id = 1
        # m = Mock(return_value = [1,2,3,4])
        x = parser.get_img_ann(img_id)
        print(x)


# @patch('torch_object_detection.parser.parser_open_images.OpenImagesParser._load_annotations')
# def test(self, load_init)
# Open("kahlua).load_annotations("kahlua)
# load_init.assert_called()
