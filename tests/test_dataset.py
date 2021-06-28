import numpy as np
from typing_extensions import OrderedDict
from unittest import TestCase 
from torch_object_detection.dataset.open_images_dataset import OpenImagesDataset
def test_len_dataset():
    path = "/home/david/fiftyone/open-images-v6"
    splits = ('train','validation')
    dataset_cls = OpenImagesDataset
    datasets = OrderedDict()
    message = "Images in dataset dont match"
    for s in splits:
        datasets[s] = dataset_cls(path, s) 
    assert datasets['train'].__len__() == 1000     
    assert datasets['validation'].__len__() == 200    

def test_input_image_size():

def test_augmentations()

    # TestCase.assertEqual(, 1000, message)



def test_shape():
    pass

