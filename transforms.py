import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import functional as F

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def clip_boxes_(boxes, img_size):
    height, width = img_size
    clip_upper = np.array([height, width] * 2, dtype=boxes.dtype)
    np.clip(boxes, 0, clip_upper, out=boxes)


def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size


class RandomResizePad:
    def __init__(
        self,
        target_size: int,
        scale: tuple = (0.1, 2.0),
        interpolation: str = "random",
        fill_color: tuple = (0, 0, 0),
    ):
        self.target_size = _size_tuple(target_size)
        self.scale = scale
        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.fill_color = fill_color

    def _get_params(self, img):
        # Select a random scale factor.
        scale_factor = random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]

        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.size
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * random.uniform(0, 1))
        return scaled_h, scaled_w, offset_y, offset_x, img_scale

    def __call__(self, img, anno: dict):
        scaled_h, scaled_w, offset_y, offset_x, img_scale = self._get_params(img)

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        img = img.resize((scaled_w, scaled_h), interpolation)
        right, lower = min(scaled_w, offset_x + self.target_size[1]), min(
            scaled_h, offset_y + self.target_size[0]
        )
        img = img.crop((offset_x, offset_y, right, lower))
        new_img = Image.new(
            "RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color
        )
        new_img.paste(img)  # pastes at 0,0 (upper-left corner)

        if "boxes" in anno:
            bbox = anno["boxes"]  # for convenience, modifies in-place
            bbox[:, :4] *= img_scale
            box_offset = np.stack([offset_y, offset_x] * 2)
            bbox -= box_offset
            bbox_bound = (
                min(scaled_h, self.target_size[0]),
                min(scaled_w, self.target_size[1]),
            )
            clip_boxes_(
                bbox, bbox_bound
            )  # crop to bounds of target image or letter-box, whichever is smaller
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno["boxes"] = bbox[valid_indices, :]
            anno["labels"] = anno["labels"][valid_indices]

        anno["img_scale"] = 1.0 / img_scale  # back to original

        return new_img, anno


class ToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        bbox = target.get("boxes")
        cls = target.get("labels")
        bbox = torch.as_tensor(bbox, dtype=torch.float64)
        cls = torch.as_tensor(cls, dtype=torch.int64)
        target.update(boxes=bbox, labels=cls)
        # print("target", target)
        return image, target


class MyCompose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, annotations: dict):
        for t in self.transforms:
            img, annotations = t(img, annotations)
        return img, annotations


def resolve_fill_color(fill_color, img_mean=IMAGENET_DEFAULT_MEAN):
    if isinstance(fill_color, tuple):
        assert len(fill_color) == 3
        fill_color = fill_color
    else:
        try:
            int_color = int(fill_color)
            fill_color = (int_color,) * 3
        except ValueError:
            assert fill_color == "mean"
            fill_color = tuple([int(round(255 * x)) for x in img_mean])
    return fill_color


def transforms_train(
    img_size,
    interpolation="random",
    fill_color="mean",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    fill_color = resolve_fill_color(fill_color, mean)
    image_tfl = [
        RandomResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=fill_color
        ),
        ToTensor(),
    ]  # tf.resize
    # return T.Compose(image_tfl)
    image_tf = MyCompose(image_tfl)
    return image_tf


def transforms_val(
    img_size,
    interpolation="bilinear",
    fill_color="mean",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
):
    fill_color = resolve_fill_color(fill_color, mean)
    image_tfl = [
        RandomResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=fill_color
        ),
        ToTensor(),
    ]
    # return T.Compose(image_tfl)
    image_tf = MyCompose(image_tfl)
    return image_tf
