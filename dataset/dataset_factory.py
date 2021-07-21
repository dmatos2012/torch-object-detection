from dataset.open_images_dataset import OpenImagesDataset
from typing_extensions import OrderedDict


def create_dataset(root, splits=("train", "validation")):
    dataset_cls = OpenImagesDataset
    datasets = OrderedDict()
    if isinstance(splits, str):
        dataset = dataset_cls(root, splits)
        return dataset
    for s in splits:
        datasets[s] = dataset_cls(root, s)
    datasets = list(datasets.values())
    return datasets
