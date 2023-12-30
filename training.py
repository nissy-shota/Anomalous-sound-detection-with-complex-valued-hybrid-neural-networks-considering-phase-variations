# Standard library imports.
import datetime
import glob
import os
import random
from itertools import chain

# Related third party imports.
import hydra
import joblib
import numpy as np
import scipy.stats
import torch
import torch.utils.data
from dotenv import load_dotenv
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_REPO_URL,
    MLFLOW_RUN_NAME,
    MLFLOW_SOURCE_NAME,
    MLFLOW_USER,
)
from omegaconf import DictConfig
from scipy.special import softmax
from torch import optim
from torch.utils.data.dataset import Subset
from torchinfo import summary


from experiments import mlflow_writer
from datamodules.dataset import FeatureExtractedToyADMOS
from datamodules.dataloader import get_dataloader
from preprocessing.feature_extraction import LoadMultiLabelFeatures
from metrics import anomaly_socre_metric


from utils import util_dirs, util_files, util_models, util_optimizers

# String constant: "cuda:0" or "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def torch_fix_seed(seed=44):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def training(model, data_loader, epoch: int, optimizer, writer, scheduler=None):
    """
    Perform training
    """
    model.train()  # training mode
    train_loss = 0.0
    for data, label in data_loader:
        data = data.to(DEVICE).type(torch.complex64)
        label = label.to(DEVICE).long()
        optimizer.zero_grad()  # reset gradient
        loss = model.get_loss(data, label)
        loss.backward()  # backpropagation
        train_loss += loss.item()
        optimizer.step()  # update paramerters

    if scheduler is not None:
        scheduler.step()  # update learning rate

    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(DEVICE).type(torch.complex64)
            label = label.to(DEVICE).long()
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    training_loss = train_loss / len(data_loader)
    training_accuracy = correct / total

    writer.log_metric_step("training loss", training_loss, step=epoch)
    writer.log_metric_step("training accuracy", training_accuracy, step=epoch)

    print("loss: {:.6f} - ".format(training_loss), end="")
    print(
        "accuracy: {:.6f}% ({}/{})".format(
            100 * float(training_accuracy), correct, total
        ),
    )


def validation(model, data_loader, epoch, writer):
    """
    Perform validation
    """
    model.eval()  # validation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(DEVICE).type(torch.complex64)
            label = label.to(DEVICE).long()
            loss = model.get_loss(data, label)
            val_loss += loss.item()

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    validation_loss = val_loss / len(data_loader)
    validation_accuracy = correct / total

    writer.log_metric_step("validation loss", validation_loss, step=epoch)
    writer.log_metric_step("validation accuracy", validation_accuracy, step=epoch)

    print("loss: {:.6f} - ".format(validation_loss), end="")
    print(
        "accuracy: {:.6f}% ({}/{})".format(
            100 * float(validation_accuracy), correct, total
        ),
    )


@hydra.main(version_base=None, config_path="configures/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    torch_fix_seed()
    os.makedirs(cfg.data_structures.artifacts_directory, exist_ok=True)
    os.makedirs(cfg.data_structures.model_directory, exist_ok=True)

    dir_list = util_files.get_machine_name_path(
        dataset_dir=cfg.data_structures.dataset_path
    )

    for idx, target_dir in enumerate(dir_list):
        machine_type = os.path.split(target_dir)[1]

        if machine_type not in cfg.dataset.target_machine_types:
            continue

        # Start recording and managing experiments
        EXPERIMENT_NAME = machine_type
        writer = mlflow_writer.MlflowWriter(
            EXPERIMENT_NAME, cfg, cfg.training.script_file_name
        )

        writer.create_new_run(writer.tags)
        writer.log_params_from_omegaconf_dict(cfg)

        print(f"Features: {cfg.preprocessing.name}")
        print(f"Model: {cfg.models.model_name}")
        print(f"Dataloader: {cfg.dataloader.name}")

        unique_case_ids = np.unique(
            util_files.get_machine_case_id(target_dir=target_dir, mode="train")
        )

        print(unique_case_ids)
        machine_case_ids_file_path = (
            f"{cfg.data_structures.model_directory}/machine_case_id_{machine_type}.pkl"
        )
        joblib.dump(unique_case_ids, machine_case_ids_file_path)

        transform = LoadMultiLabelFeatures(is_eval=False, cfg=cfg)
        ToyADMOS_dataset = FeatureExtractedToyADMOS(
            unique_case_ids=unique_case_ids,
            target_dir=target_dir,
            cfg=cfg,
            transform=transform,
        )

        data_loader = {"train": None, "val": None, "eval_train": None}
        (
            data_loader["train"],
            data_loader["val"],
            data_loader["eval_train"],
        ) = get_dataloader(
            dataset=ToyADMOS_dataset, config=cfg, machine_type=machine_type
        )

        model = util_models.get_model(
            num_classes=len(unique_case_ids), config=cfg, DEVICE=DEVICE
        )
        params = 0
        for p in model.parameters():
            if p.requires_grad:
                params += p.numel()
        print(f"number of model parameter is {params}")

        optimizer, scheduler = util_optimizers.get_optimizer_scheduler(
            model, config=cfg
        )  # optional

        for epoch in range(1, cfg.training.epochs + 1):
            now = datetime.datetime.now()
            now_str = now.strftime("%Y/%m/%d %H:%M:%S")
            print("{} Epoch {:2d} Train: ".format(now_str, epoch), end="")
            training(
                model=model,
                data_loader=data_loader["train"],
                epoch=epoch,
                optimizer=optimizer,
                writer=writer,
                scheduler=scheduler,  # optional
            )
            now = datetime.datetime.now()
            now_str = now.strftime("%Y/%m/%d %H:%M:%S")
            print("{} Epoch {:2d} Valid: ".format(now_str, epoch), end="")
            validation(
                model=model, data_loader=data_loader["val"], epoch=epoch, writer=writer
            )

        del ToyADMOS_dataset, data_loader

        if cfg.metrics.metric_pdf == "gaussian":
            loc, scale = anomaly_socre_metric.fit_gaussian_dist(
                model=model,
                target_dir=target_dir,
                mode="train",
                config=cfg,
                device=DEVICE,
            )
            writer.log_metric("loc", loc)
            writer.log_metric("scale", scale)
        else:
            raise ValueError("Input gaussian!")

        util_models.save_model(
            model,
            model_dir=cfg.data_structures.model_directory,
            machine_type=machine_type,
        )

        artifact_files = glob.glob(
            f"{os.path.abspath(cfg.data_structures.model_directory)}/*{machine_type}*"
        )
        for artifact_file in artifact_files:
            writer.log_artifact(artifact_file)
        writer.set_terminated()


if __name__ == "__main__":
    main()
