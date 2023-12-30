from omegaconf import DictConfig
import torch
import os
import sys


def get_model(num_classes, config: DictConfig, DEVICE):

    n_fft = config.preprocessing.feature.n_fft
    n_fft_dim = int(1 + n_fft / 2)

    num_features = config.preprocessing.feature.n_frames * n_fft_dim

    if config.models.model_name == "MLP":

        from models.mlp import SimpleMLP

        model = SimpleMLP(
            num_features=num_features,
            num_classes=num_classes,
            num_hidden=config.models.num_hidden,
        ).to(DEVICE)
        return model
    elif config.models.model_name == "CNN":
        from models.cnn import ComplexCNN

        model = ComplexCNN(
            num_classes=num_classes,
            cfg=config,
        ).to(DEVICE)
        return model
    elif config.models.model_name == "ResNet":
        if config.models.model_depth == 18:

            from models.resnet import resnet18

            model = resnet18(num_classes=num_classes).to(DEVICE)
            return model

        elif config.models.model_depth == 50:
            from models.resnet import resnet50

            model = resnet50(num_classes=num_classes).to(DEVICE)
            return model
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Input CNN or MLP")


def load_model(machine_type, n_machine_id, config: DictConfig, device):
    """
    Load model file
    """
    model_file = "{model}/model_{machine_type}.hdf5".format(
        model=config.data_structures.model_directory, machine_type=machine_type
    )
    if not os.path.exists(model_file):
        print("{} model not found ".format(machine_type))
        sys.exit(-1)
    n_fft = config.preprocessing.feature.n_fft
    n_fft_dim = int(1 + n_fft / 2)
    num_features = config.preprocessing.feature.n_frames * n_fft_dim

    if config.models.model_name == "MLP":
        from models.mlp import SimpleMLP

        model = SimpleMLP(
            num_features=num_features,
            num_classes=n_machine_id,
            num_hidden=config.models.num_hidden,
        ).to(device)
    elif config.models.model_name == "CNN":
        from models.cnn import ComplexCNN

        model = ComplexCNN(
            num_classes=n_machine_id,
            cfg=config,
        ).to(device)
    elif config.models.model_name == "ResNet":
        if config.models.model_depth == 18:
            from models.resnet import resnet18

            model = resnet18(num_classes=n_machine_id).to(device)
            return model
        elif config.models.model_depth == 50:
            from models.resnet import resnet50

            model = resnet50(num_classes=n_machine_id).to(device)
            return model
        else:
            raise NotImplementedError()
    else:
        raise ValueError("Input CNN or MLP")

    model.eval()
    model.load_state_dict(torch.load(model_file))

    return model


def save_model(model, model_dir, machine_type):
    """
    Save PyTorch model.
    """

    model_file_path = "{model}/model_{machine_type}.hdf5".format(
        model=model_dir, machine_type=machine_type
    )
    torch.save(model.state_dict(), model_file_path)
    print("save_model -> %s" % (model_file_path))
