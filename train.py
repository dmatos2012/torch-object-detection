from pathlib import Path

import cv2
import numpy as np
import torchvision
from typing_extensions import OrderedDict

from dataset.loader import create_loader
from dataset.open_images_dataset import OpenImagesDataset
from fasterrcnn import get_model

interpolation = "bilinear"
fill_color = "mean"
num_workers = 2


def create_datasets_and_loaders(transform_train_fn=None):
    input_size = 224  # input of image
    batch_size = 2
    num_workers = 2
    root = Path("/home/david/fiftyone/open-images-v6")
    dataset_train, dataset_val = create_dataset(root)
    print(dataset_train.__len__())
    print(dataset_val.__len__())
    visualize_input(dataset_train)

    loader_train = create_loader(
        dataset_train,
        input_size,
        batch_size,
        interpolation=interpolation,
        fill_color=fill_color,
        num_workers=num_workers,
        is_training=True,
    )

    loader_val = create_loader(
        dataset_val,
        input_size,
        batch_size,
        interpolation=interpolation,
        fill_color=fill_color,
        num_workers=num_workers,
        is_training=False,
    )
    return loader_train, loader_val


def visualize_input(dataset):
    pil_img, target = dataset.__getitem__(1)
    bboxes = target["boxes"]
    color = (255, 0, 0)
    thickness = 2

    for bbox in bboxes:
        bbox = list(map(lambda x: int(x), bbox))
        print(bbox)
        cv_img = np.array(pil_img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        startp = (bbox[0], bbox[1])
        endp = (bbox[2], bbox[3])
        cv2.rectangle(cv_img, startp, endp, color, thickness)
    cv2.imwrite("hi.jpg", cv_img)
    print("img created")


def create_dataset(root, splits=("train", "validation")):
    dataset_cls = OpenImagesDataset
    datasets = OrderedDict()
    for s in splits:
        datasets[s] = dataset_cls(root, s)
    datasets = list(datasets.values())
    return datasets


# print(loader)
model = get_model()
loader_train, loader_val = create_datasets_and_loaders()


for input, target in loader_train:
    # print(input[0].shape)
    # print(input[1].shape)
    output = model(input, target)
    torchvision.utils.save_image(
        list(input), "train-batch.jpg", padding=0, normalize=True
    )
    print("saved image loader")
