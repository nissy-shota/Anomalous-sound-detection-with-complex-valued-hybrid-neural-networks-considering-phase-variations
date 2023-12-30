import torch
import torch.nn as nn
import torch.nn.functional as F

from complexmodules import complex_activation_function


class ComplexMLP(nn.Module):

    """
    This is the simplest MLP baseline. BatchNorm and Dropout are not implemented.
    The reason for this is to perform ablation studies with complex-valued-MLP.
    """

    def __init__(self, num_features: int, num_classes: int, num_hidden: int):

        super(ComplexMLP, self).__init__()
        self.fc1 = nn.Linear(num_features, num_hidden).to(torch.complex64)
        self.fc2 = nn.Linear(num_hidden, num_hidden).to(torch.complex64)
        self.classifier = nn.Linear(num_hidden, num_classes).to(torch.complex64)

        self.criterion = nn.NLLLoss()
        self.activation = complex_activation_function.real_imaginary_relu

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = self.activation(self.fc1(inputs))
        x = self.activation(self.fc2(x))
        h = self.classifier(x)
        h = torch.square(torch.abs(h))
        x = F.log_softmax(h, dim=1)
        return x

    def get_loss(self, inputs, labels):

        output = self.forward(inputs)
        loss = self.criterion(output, labels)

        return loss
