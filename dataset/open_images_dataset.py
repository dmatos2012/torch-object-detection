from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from torch_object_detection.parser.parser_open_images import OpenImagesParser


class OpenImagesDataset(Dataset):
    def __init__(self, root, splits, transform=None):
        self.root = root
        if isinstance(root, str):
            self.root = Path(root)
        self.splits = splits
        self.data_dir = self.root / self.splits / "data"
        self.imgs = sorted((self.root / self.splits / "data").glob("*.jpg"))
        # self.labels = sorted((Path(root) / "labels" / "train2017").glob('*.txt'))
        self.ann_file = self.root / self.splits / "labels.json"
        self.parser = OpenImagesParser(self.ann_file)
        self._transform = transform

    def __getitem__(self, index):

        img_info = self.parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info["width"], img_info["height"]))
        if self.parser.has_labels:
            ann = self.parser.get_ann_info(index)
            target.update(ann)

        img_path = self.data_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img, target = self.transform(img, target)
        # print(target)
        return img, target

    def __len__(self):
        return len(self.parser.img_ids)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


# root = Path().absolute() / "coco128"
# root = Path("/home/david/fiftyone/open-images-v6")
# dataset = OpenImagesDataset(root)
# dataset.__getitem__(0)
# print(dataset.__len__())
# input_size = (3, 512, 512) # input of image
# batch_size = 2
# num_workers = 2
# loader = create_loader(dataset, input_size, batch_size,
#         num_workers,
#         True)
