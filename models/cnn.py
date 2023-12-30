import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from complexmodules import complex_activation_function
from complexmodules import complex_functional as CF
from complexmodules import complex_modules


class ComplexCNN(nn.Module):
    def __init__(self, num_classes: int, cfg: DictConfig):
        super(ComplexCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=cfg.preprocessing.conv1_in_channels,
            out_channels=cfg.models.conv1_out_channels,
            kernel_size=cfg.models.conv1_kernel_size,
        ).to(torch.complex64)

        self.conv2 = nn.Conv2d(
            in_channels=cfg.models.conv1_out_channels,
            out_channels=cfg.models.conv2_out_channels,
            kernel_size=cfg.models.conv2_kernel_size,
        ).to(torch.complex64)

        self.conv3 = nn.Conv2d(
            in_channels=cfg.models.conv2_out_channels,
            out_channels=cfg.models.conv3_out_channels,
            kernel_size=cfg.models.conv3_kernel_size,
        ).to(torch.complex64)

        self.fc1 = nn.Linear(cfg.preprocessing.linear_in, cfg.models.num_hidden).to(
            torch.complex64
        )
        self.fc2 = nn.Linear(cfg.models.num_hidden, cfg.models.num_hidden).to(
            torch.complex64
        )
        self.classifier = nn.Linear(cfg.models.num_hidden, num_classes).to(
            torch.complex64
        )

        self.activation = complex_activation_function.real_imaginary_relu
        self.criterion = nn.NLLLoss()

        self.bn1 = complex_modules.ComplexBatchNorm2d(cfg.models.conv1_out_channels)
        self.bn2 = complex_modules.ComplexBatchNorm2d(cfg.models.conv2_out_channels)
        self.bn3 = complex_modules.ComplexBatchNorm2d(cfg.models.conv3_out_channels)
        self.bn4 = complex_modules.ComplexBatchNorm1d(cfg.models.num_hidden)
        self.bn5 = complex_modules.ComplexBatchNorm1d(cfg.models.num_hidden)

        self.pool1 = complex_modules.ComplexMaxPool2d(2, 2).to(torch.complex64)
        self.pool2 = complex_modules.ComplexMaxPool2d(2, 2).to(torch.complex64)
        self.pool3 = complex_modules.ComplexMaxPool2d(2, 2).to(torch.complex64)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = self.pool1(self.activation(self.bn1(self.conv1(inputs))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.activation(self.bn4(self.fc1(x)))
        x = self.activation(self.bn5(self.fc2(x)))
        h = self.classifier(x)  # for complex
        h = torch.square(torch.abs(h))
        x = F.log_softmax(h, dim=1)
        return x

    def get_loss(self, inputs, labels):

        output = self.forward(inputs)
        loss = self.criterion(output, labels)

        return loss


class ComplexCNNWithoutDropout(nn.Module):
    def __init__(self, num_classes: int, num_hidden: int, cfg: DictConfig):
        super(ComplexCNNWithoutDropout, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=cfg.conv1_out_channels,
            kernel_size=cfg.conv1_kernel_size,
        ).to(torch.complex64)

        self.conv2 = nn.Conv2d(
            in_channels=cfg.conv1_out_channels,
            out_channels=cfg.conv2_out_channels,
            kernel_size=cfg.conv2_kernel_size,
        ).to(torch.complex64)

        self.fc1 = nn.Linear(26000, num_hidden).to(torch.complex64)
        self.fc2 = nn.Linear(num_hidden, num_hidden).to(torch.complex64)
        self.classifier = nn.Linear(num_hidden, num_classes).to(torch.complex64)

        self.activation = complex_activation_function.real_imaginary_relu
        self.criterion = nn.NLLLoss()

        self.bn1 = complex_modules.ComplexBatchNorm2d(cfg.conv1_out_channels)
        self.bn2 = complex_modules.ComplexBatchNorm2d(cfg.conv2_out_channels)
        self.bn3 = complex_modules.ComplexBatchNorm1d(num_hidden)
        self.bn4 = complex_modules.ComplexBatchNorm1d(num_hidden)

        self.pool1 = complex_modules.ComplexMaxPool2d(2, 2).to(torch.complex64)
        self.pool2 = complex_modules.ComplexMaxPool2d(2, 2).to(torch.complex64)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = self.pool1(self.activation(self.bn1(self.conv1(inputs))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.activation(self.bn3(self.fc1(x)))
        x = self.activation(self.bn4(self.fc2(x)))
        h = self.classifier(x)  # for complex
        h = torch.square(torch.abs(h))
        x = F.log_softmax(h, dim=1)
        return x

    def get_loss(self, inputs, labels):

        output = self.forward(inputs)
        loss = self.criterion(output, labels)

        return loss
