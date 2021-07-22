from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from dataset.dataset_factory import create_dataset
from dataset.loader import create_loader
from easydict import EasyDict
from fasterrcnn import get_model

# from utils.coco_evaluate import evaluate
from utils.load_config import load_yaml

yaml_config = "config.yaml"
config = load_yaml(yaml_config)
config = EasyDict(config)
ckpt_name = "model_ckpt_epoch8.pth"
ckpt_path = Path() / "output" / ckpt_name
root = Path(config.root)
dataset_val = create_dataset(root, splits=("validation"))
config.batch_size = 2
loader_val = create_loader(
    dataset_val,
    config.input_size,
    config.batch_size,
    interpolation=config.aug.interpolation,
    fill_color=config.aug.fill_color,
    num_workers=config.num_workers,
    is_training=False,
)


# From https://pytorch.org/vision/master/auto_examples/plot_visualization_utils.html
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()


def load_ckpt(ckpt_path, device):
    model = get_model()
    model.cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.opt.lr,
        momentum=config.opt.momentum,
        weight_decay=config.opt.weight_decay,
    )
    try:
        checkpoint = torch.load(str(ckpt_path))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"{ckpt_path.name} loaded successfully")
        return model
    except FileNotFoundError:
        print(f"could not find {ckpt_path}. Please enter valid ckpt")


# grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    new_model = load_ckpt(ckpt_path, device)
    img, _ = dataset_val[2]
    pred_img = img.detach().clone()
    new_model.eval()
    with torch.no_grad():
        prediction = new_model([pred_img.to(device)])[0]  # bc unique image
        # prediction = new_model([pred_img])[0]
        img = img.mul(255).to(torch.uint8)
        boxes = prediction["boxes"]
        labels = prediction["labels"]
        score_thr = 0.29
        res_img = torchvision.utils.draw_bounding_boxes(
            image=img, boxes=prediction["boxes"][prediction["scores"] > score_thr]
        )
        show(res_img)
        # evaluate(new_model, loader_val, device=device)
        print(prediction)
