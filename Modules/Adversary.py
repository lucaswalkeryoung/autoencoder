# --------------------------------------------------------------------------------------------------
# ----------------------------------------- Latent Decoder -----------------------------------------
# --------------------------------------------------------------------------------------------------
from torch.nn import functional as functional
from torch import nn as networks

from Modules.Network import Network

from torch import Tensor
from torch import sigmoid
import torch


# --------------------------------------------------------------------------------------------------
# --------------------------------------- CLASS :: Adversary ---------------------------------------
# --------------------------------------------------------------------------------------------------
class Adversary(Network):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self):
        super(Adversary, self).__init__(name="Adversary")

        self.conv1 = networks.Conv2d(3,  16, kernel_size=3, stride=1, padding=1)
        self.norm1 = networks.BatchNorm2d(16)
        self.conv2 = networks.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.norm2 = networks.BatchNorm2d(16)
        self.conv3 = networks.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm3 = networks.BatchNorm2d(16)
        self.conv4 = networks.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.norm4 = networks.BatchNorm2d(16)
        self.conv5 = networks.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm5 = networks.BatchNorm2d(16)
        self.conv6 = networks.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.norm6 = networks.BatchNorm2d(16)
        self.conv7 = networks.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.norm7 = networks.BatchNorm2d(16)
        self.conv8 = networks.Conv2d(16,  1, kernel_size=3, stride=2, padding=1)

        self.final = networks.AdaptiveMaxPool2d((1, 1))

        self.funct = networks.LeakyReLU(0.2)
        self.noise = networks.Dropout(0.000)


    # ------------------------------------------------------------------------------------------
    # ----------------------------- METHOD :: Forward Propagation ------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:

        x = self.funct(self.norm1(self.conv1(self.noise(x))))
        x = self.funct(self.norm2(self.conv2(self.noise(x))))
        x = self.funct(self.norm3(self.conv3(self.noise(x))))
        x = self.funct(self.norm4(self.conv4(self.noise(x))))
        x = self.funct(self.norm5(self.conv5(self.noise(x))))
        x = self.funct(self.norm6(self.conv6(self.noise(x))))
        x = self.funct(self.norm7(self.conv7(self.noise(x))))
        x = self.funct(self.conv8(self.noise(x)))

        return sigmoid(self.final(x).view(x.size(0), -1))


    # ------------------------------------------------------------------------------------------
    # ---------------------------- METHOD :: Compute Loss Function -----------------------------
    # ------------------------------------------------------------------------------------------
    def loss(self, predictions: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:

        correctness = labels * predictions + (1 - labels) * (1 - predictions)
        certainty   = torch.abs(predictions - 0.5) * 2

        negative = 0 + (correctness * certainty).mean().detach()
        positive = 1 - (correctness * certainty).mean()

        return positive * 10, negative * 10

