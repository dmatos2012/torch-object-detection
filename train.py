import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from dataset.dataset_factory import create_dataset
from dataset.loader import create_loader
from fasterrcnn import get_model
from utils import torch_utils
from utils.coco_evaluate import evaluate
from utils.load_config import load_yaml

yaml_config = "config.yaml"
hyp = load_yaml(yaml_config)
output_dir = os.path.join(os.getcwd(), "output")
sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "mAP_050"},
    "parameters": {
        "lr": {"values": [0.0005, 0.005, 0.010]},
        "weight_decay": {"values": [0.0005, 0.005]},
        "batch_size": {"values": [4, 8, 12]},
        "input_size": {"values": [128, 224, 448]},
        "num_epochs": {"values": [10, 15, 20]},
    },
}
#   opt.opt:
#     values: ["SGD", "Adam"]

# Now initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="open-images-detection")


def create_datasets_and_loaders(input_size, batch_size, transform_train_fn=None):
    # input_size = 224  # input of image
    # batch_size = 2
    root = Path(hyp["root"])
    dataset_train, dataset_val = create_dataset(root)
    print(dataset_train.__len__())
    print(dataset_val.__len__())
    # visualize_input(dataset_train)

    loader_train = create_loader(
        dataset_train,
        input_size,
        batch_size,
        interpolation=hyp["interpolation"],
        fill_color=hyp["fill_color"],
        num_workers=hyp["num_workers"],
        is_training=True,
    )

    loader_val = create_loader(
        dataset_val,
        input_size,
        batch_size,
        interpolation=hyp["interpolation"],
        fill_color=hyp["fill_color"],
        num_workers=hyp["num_workers"],
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


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = get_model()
# loader_train, loader_val = create_datasets_and_loaders()
# model.to(device)
# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
#     params,
#     lr=hyp["lr"],
#     momentum=hyp["momentum"],
#     weight_decay=hyp["weight_decay"],
# )
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer, step_size=hyp["step_size"], gamma=hyp["gamma"]
# )
#


def train(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = torch_utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter(
        "lr", torch_utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch_utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor
        )

    for batch_idx, (inputs, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        inputs = list(img.to(device) for img in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = torch_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if batch_idx % print_freq == 0:
            wandb.log({"Train loss": loss_value})
        # torchvision.utils.save_image(list(inputs),
        # "train%s-batch.jpg" %batch_idx, padding=0, normalize=True)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        wandb.log({"lr": optimizer.param_groups[0]["lr"]})

    return metric_logger, loss_value


def validate(model, optimizer, data_loader, device, print_freq):
    # https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333
    model.train()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = list(img.to(device) for img in inputs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(inputs, targets)
            loss_dict_reduced = torch_utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            if batch_idx % print_freq == 0:
                wandb.log({"Validation loss": loss_value})
            # torchvision.utils.save_image(list(inputs),
            # "train%s-batch.jpg" %batch_idx, padding=0, normalize=True)
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

    return loss_value


print_freq = 10


def run_training():
    with wandb.init(config=hyp, entity="dmatos"):
        config = wandb.config
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_model()
        wandb.watch(model)
        loader_train, loader_val = create_datasets_and_loaders(
            config["input_size"], config["batch_size"]
        )
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config["step_size"], gamma=config["gamma"]
        )
        for epoch in range(config["num_epochs"]):
            # run_url = wandb.run.get_url()
            _, train_loss_epoch = train(
                model, optimizer, loader_train, device, epoch, print_freq=print_freq
            )
            val_loss_epoch = validate(
                model, optimizer, loader_val, device, print_freq=print_freq
            )
            wandb.log(
                {
                    "Epoch Train Loss": train_loss_epoch,
                    "Epoch Validation Loss": val_loss_epoch,
                }
            )
            # losses = [[train_loss, val_loss]]
            # table = wandb.Table(data=losses, columns=["Train Loss", "Validation Loss"])

            if ((epoch + 1) % 5 == 0) or epoch == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    output_dir
                    + "/"
                    + "model_ckpt_epoch%s_sweep%s.pth" % (epoch, sweep_id),
                )
                _, coco_stats = evaluate(model, loader_val, device=device)
                wandb.log({"mAP_050": coco_stats[1]})

            lr_scheduler.step()


count = 5
wandb.agent(sweep_id, function=run_training, count=count)
