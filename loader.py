import torch
import torch.utils
import torch.utils.data

from torch_object_detection.transforms import *


def my_collate_fn(batch):
    return tuple(zip(*batch))


def create_loader(
    dataset,
    input_size,
    batch_size,
    interpolation="bilinear",
    fill_color="mean",
    num_workers=2,
    is_training=True,
    transform_fn=None,
    collate_fn=None,
):
    if is_training:
        transform = transforms_train(
            input_size,
            interpolation=interpolation,
            fill_color=fill_color,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    else:
        transform = transforms_val(
            input_size,
            interpolation=interpolation,
            fill_color=fill_color,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_MEAN,
        )
    dataset.transform = transform
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        collate_fn=my_collate_fn,
    )
    return loader
