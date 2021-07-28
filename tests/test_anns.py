from unittest import TestCase
from unittest.mock import patch

import numpy as np
from dataset.parser_open_images import OpenImagesParser
from numpy.testing import assert_array_equal


class TestNetwork(TestCase):
    # @patch('pycocotools.coco.COCO.getAnnIds')
    @patch("pycocotools.coco.COCO")
    @patch("dataset.parser_open_images.OpenImagesParser._load_annotations")
    def test_img_anno(self, _, mock_coco):
        # input bbox in xywh format.
        ann = [
            dict(
                id=10,
                image_id=1,
                category_id=160,
                bbox=np.array(([10.0, 20.0, 30.0, 40.0])),
                area=1200.0,
                iscrowd=None,
            )
        ]
        # fix the category_id one, since now its forced to be either 160 or 96(my two classes)
        parser = OpenImagesParser("label.json")
        parser.coco = mock_coco
        parser.coco.loadAnns.return_value = ann
        # parser._load_annotations("kahlua.json")
        img_id = 1
        result = parser.get_img_ann(img_id)
        ann_dict = ann[0]
        x1 = ann_dict["bbox"][0]
        y1 = ann_dict["bbox"][1]
        x2 = x1 + ann_dict["bbox"][2]
        y2 = y1 + ann_dict["bbox"][3]
        bbox_to_xyxy = np.array(([[x1, y1, x2, y2]]))
        # TODO: assert label as well
        msg = "Bounding boxes were not transformed properly"
        assert_array_equal(result["boxes"], bbox_to_xyxy, msg)
