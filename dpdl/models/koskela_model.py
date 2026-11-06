import logging
import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms

log = logging.getLogger(__name__)

class KoskelaNet(nn.Module):
    """
    This is the network used in the paper "Learning Rate Adaptation for Federated and
    Differentially Private Learning" (Koskela et al., 2019)

    https://arxiv.org/pdf/1809.03832.pdf

    It consists of these two parts that are stacked one after another:

    Net1(
        (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
        (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    )

    Net2(
        (relu): ReLU()
        (linears): ModuleList(
            (0): Linear(in_features=1600, out_features=500, bias=False)
            (1): Linear(in_features=500, out_features=500, bias=False)
        )
        (final_fc): Linear(in_features=500, out_features=10, bias=False)
    )

    We are otherwise using it as-is, but ignoring the custom Linear layer that they use:
    https://github.com/DPBayes/ADADP/blob/master/CIFAR_tests/linear.py

    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(1600, 500, bias=False)
        self.fc2 = nn.Linear(500, 500, bias=False)
        self.final_fc = nn.Linear(500, num_classes, bias=False)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final_fc(x)

        return x

    def get_classifier(self):
        return self.final_fc

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def save_model(self, fpath):
        torch.save(self.state_dict(), fpath)

