from typing import Dict

import torch
from omegaconf import DictConfig
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset


def get_dataloader(dataset, config: DictConfig, machine_type):
    """
    Make dataloader from dataset for training.
    """
    dataset_size = len(dataset)
    train_size = int(len(dataset) * (1.0 - config.training.validation_split))
    valid_size = dataset_size - train_size

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(44)
    )
    if machine_type == "ToyConveyor":
        train_dataset = Subset(
            train_dataset, list(range(0, config.dataloader.conveyor_train_size))
        )
        valid_dataset = Subset(
            valid_dataset, list(range(0, config.dataloader.conveyor_valid_size))
        )
    else:
        train_dataset = Subset(
            train_dataset, list(range(0, config.dataloader.train_size))
        )
        valid_dataset = Subset(
            valid_dataset, list(range(0, config.dataloader.valid_size))
        )

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        drop_last=True,
        num_workers=config.dataloader.num_workers,
    )

    data_loader_val = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.dataloader.num_workers,
    )

    # dataloader of training data for evaluation only
    data_loader_eval_train = torch.utils.data.DataLoader(
        Subset(dataset, list(range(0, train_size))),
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.dataloader.num_workers,
    )

    return data_loader_train, data_loader_val, data_loader_eval_train
