# --------------------------------------------------------------------------------------------------
# ----------------------------------------- Latent Decoder -----------------------------------------
# --------------------------------------------------------------------------------------------------
from torch.nn import functional as functional
from torch import nn as networks

from Modules.Network import Network

from torch import Tensor
from torch import sigmoid


# --------------------------------------------------------------------------------------------------
# ---------------------------------------- CLASS :: Stylist ----------------------------------------
# --------------------------------------------------------------------------------------------------
class Decoder(Network):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self):
        super(Decoder, self).__init__(name="Decoder")

        self.conv1 = networks.ConvTranspose2d(16, 16, kernel_size=5, stride=1, padding=1)
        self.norm1 = networks.BatchNorm2d(16)
        self.conv2 = networks.ConvTranspose2d(16, 16, kernel_size=5, stride=1, padding=1)
        self.norm2 = networks.BatchNorm2d(16)
        self.conv3 = networks.ConvTranspose2d(16, 16, kernel_size=5, stride=1, padding=1)
        self.norm3 = networks.BatchNorm2d(16)
        self.conv4 = networks.ConvTranspose2d(16,  3, kernel_size=5, stride=1, padding=1)

        self.noise = networks.Dropout(0.000)
        self.funct = networks.LeakyReLU(0.1)


    # ------------------------------------------------------------------------------------------
    # ----------------------------- METHOD :: Forward Propagation ------------------------------
    # ------------------------------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:

        x = self.funct(self.norm1(self.conv1(x)))
        x = self.funct(self.norm2(self.conv2(x)))
        x = self.funct(self.norm3(self.conv3(x)))

        return sigmoid(self.conv4(x))


    # ------------------------------------------------------------------------------------------
    # ---------------------------- METHOD :: Compute Loss Function -----------------------------
    # ------------------------------------------------------------------------------------------
    def loss(self, source: Tensor, target) -> Tensor:
        return functional.mse_loss(source, target) * 10