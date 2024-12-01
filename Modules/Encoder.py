# --------------------------------------------------------------------------------------------------
# ----------------------------------------- Latent Encoder -----------------------------------------
# --------------------------------------------------------------------------------------------------
from torch import functional as functions
from torch import nn as networks

from Modules.Network import Network

import torch


# --------------------------------------------------------------------------------------------------
# ---------------------------------------- CLASS :: Stylist ----------------------------------------
# --------------------------------------------------------------------------------------------------
class Encoder(Network):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self):
        super(Encoder, self).__init__(name="Encoder")

        self.conv1 = networks.Conv2d(3,  16, kernel_size=5, stride=1, padding=1)
        self.norm1 = networks.BatchNorm2d(16)
        self.conv2 = networks.Conv2d(16, 16, kernel_size=5, stride=1, padding=1)
        self.norm2 = networks.BatchNorm2d(16)
        self.conv3 = networks.Conv2d(16, 16, kernel_size=5, stride=1, padding=1)
        self.norm3 = networks.BatchNorm2d(16)
        self.conv4 = networks.Conv2d(16, 16, kernel_size=5, stride=1, padding=1)
        self.norm4 = networks.BatchNorm2d(16)

        self.noise = networks.Dropout(0.000)
        self.funct = networks.LeakyReLU(0.1)


    # ------------------------------------------------------------------------------------------
    # ----------------------------- METHOD :: Forward Propagation ------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.funct(self.norm1(self.conv1(self.noise(x))))
        x = self.funct(self.norm2(self.conv2(self.noise(x))))
        x = self.funct(self.norm3(self.conv3(self.noise(x))))
        x = self.funct(self.norm4(self.conv4(self.noise(x))))

        return x


