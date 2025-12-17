import os
import argparse
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch.utils.data
import torch.optim as optim
import torch

from lib.architecture import PatchClassifier
from lib.dataload import get_dataloader
from lib.eval import validation_epoch
from lib.train import train_epoch

from torch.utils.tensorboard import SummaryWriter


def get_experiment_name(patch_size, n_features):
    dt = datetime.now()
    DateTime = dt.strftime("%y%m%d%H%M%S")
    return f"{DateTime}-{patch_size}-{n_features}"


def write_exp_config(path_log: str, config: dict):
    with open(path_log, "a", encoding="utf-8") as log:
        for key, val in config.items():
            log.write(f"{key}: {val}\n")


def drop_prefix(state_dict, prefix):
    new_state_dict = OrderedDict()
    for name, w in state_dict.items():
        if name.startswith(prefix):
            name = name[len(prefix) :]
        new_state_dict[name] = w
    return new_state_dict


def run(
    train_dataloader,
    val_dataloader,
    patch_size,
    n_features,
    pre_ckpt,
    max_epoch,
    init_lr,
):

    assert (
        train_dataloader.dataset.class_to_idx == val_dataloader.dataset.class_to_idx
    ), "Class to index dictinoary differs between train and val dataloader."

    device = torch.device("cuda")

    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    experiment_name = get_experiment_name(patch_size, n_features)
    path_exp_home = f"./saved_models/{experiment_name}"
    path_best_acc_ckpt = f"{path_exp_home}/best_accuracy.pth"
    path_log = f"{path_exp_home}/exp_log.txt"
    os.makedirs(f"{path_exp_home}", exist_ok=True)
    tb_writer = SummaryWriter(f"{path_exp_home}/tensorboard")

    model = PatchClassifier(patch_size=patch_size, n_features=n_features)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    if pre_ckpt is not None:
        ckpt = torch.load(pre_ckpt)
        try:
            model.load_state_dict(ckpt["state_dict"])
        except:
            model.load_state_dict(drop_prefix(ckpt["state_dict"], "module."))

    optimizer = optim.AdamW(filtered_parameters, lr=init_lr)

    write_exp_config(
        path_log,
        {
            "class": train_dataloader.dataset.class_to_idx,
            "# trainable params": sum(params_num),
            "checkpoint": pre_ckpt,
            "lr": init_lr,
        },
    )

    best_accuracy = -1
    for epoch in range(max_epoch):

        torch.cuda.empty_cache()

        #########
        # train #
        #########
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        tb_writer.add_scalar(
            tag="Train_epoch/loss", scalar_value=train_loss, global_step=epoch
        )

        ############
        # validate #
        ############
        val_loss, val_acc = validation_epoch(model, val_dataloader, criterion, device)
        tb_writer.add_scalar(tag="Eval/loss", scalar_value=val_loss, global_step=epoch)
        tb_writer.add_scalar(tag="Eval/acc", scalar_value=val_acc, global_step=epoch)

        # save model regularly and with best valid accuracy
        state_dict = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(state_dict, path_best_acc_ckpt)

    tb_writer.close()

    return {}, {}, path_best_acc_ckpt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--n_features", type=int)
    parser.add_argument("--gpu_index", type=str)
    parser.add_argument("--dir_train_data", type=str)
    parser.add_argument("--dir_val_data", type=str)
    args = parser.parse_args()

    patch_size = args.patch_size
    n_features = args.n_features
    gpu_index = args.gpu_index
    dir_train_data = args.dir_train_data
    dir_val_data = args.dir_val_data

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

    assert torch.cuda.is_available(), "GPU not available."

    train_dataloader = get_dataloader(
        dir_train_data,
        path_augment=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lib", "augmentation.json"
        ),
        batch_size=256,
        is_train=True,
        num_workers=32,
    )

    val_dataloader = get_dataloader(
        dir_val_data,
        path_augment=None,
        batch_size=256,
        is_train=False,
        num_workers=32,
    )

    run(
        train_dataloader,
        val_dataloader,
        patch_size=patch_size,
        n_features=n_features,
        pre_ckpt=None,
        max_epoch=1000,
        init_lr=1e-4,
    )
