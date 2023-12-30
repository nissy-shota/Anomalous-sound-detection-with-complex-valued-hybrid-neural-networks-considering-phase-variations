import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import relu, leaky_relu


Tensor = torch.Tensor


def real_imaginary_relu(z):
    """_summary_
    https://github.com/pytorch/pytorch/issues/47052
    Args:
        z (_type_): _description_

    Returns:
        _type_: _description_
    """
    return relu(z.real) + 1.0j * relu(z.imag)


def complex_leaky_relu(z, negative_slope):

    return leaky_relu(z.real, negative_slope) + 1.0j * leaky_relu(
        z.imag, negative_slope
    )


def phase_amplitude_relu(z):
    return relu(torch.abs(z)) * torch.exp(1.0j * torch.angle(z))


def amplitude_phase_activation(z):
    return torch.tanh(torch.abs(z)) * torch.exp(1.0j * torch.angle(z))


def complex_softmax(input, axis=1):
    """_summary_
    complex-valued softmax without log

    Complex-valued Neural Networks with Non-parametric Activation Functions
    Eq.(36)
    https://arxiv.org/abs/1802.08026

    Args:
        input (_type_): _description_

    Returns:
        _type_: _description_
    """

    h = torch.abs(input)
    h = torch.square(h)

    return F.softmax(h, axis)


class ComplexCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ComplexCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        if torch.is_complex(inputs):
            real_loss = self.criterion(inputs.real, targets)
            imag_loss = self.criterion(inputs.imag, targets)
            return (real_loss + imag_loss) / 2
        else:
            return self.criterion(inputs, targets)
