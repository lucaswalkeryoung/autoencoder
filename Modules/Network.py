# --------------------------------------------------------------------------------------------------
# ----------------------------------- Abstract-Base Module Class -----------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn.init as initializers
import torch.nn as networks

import pathlib
import torch

from safetensors.torch import save_file
from safetensors.torch import load_file


# --------------------------------------------------------------------------------------------------
# --------------------------------------- CLASS :: Adversary ---------------------------------------
# --------------------------------------------------------------------------------------------------
class Network(networks.Module):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, name: str):
        super(Network, self).__init__()

        self.name = name
        self.path = pathlib.Path(f"./Models/{name}.safetensors")


    # ----------------------------------------------------------------------------------------------
    # ---------------------------------- METHOD :: Load the Model ----------------------------------
    # ----------------------------------------------------------------------------------------------
    def load(self) -> None:
        self.load_state_dict(load_file(self.path))


    # ----------------------------------------------------------------------------------------------
    # ---------------------------------- METHOD :: Save the Model ----------------------------------
    # ----------------------------------------------------------------------------------------------
    def save(self) -> None:
        save_file(self.state_dict(), self.path)


    # ----------------------------------------------------------------------------------------------
    # -------------------------- METHOD :: Initialize the Model's Weights --------------------------
    # ----------------------------------------------------------------------------------------------
    def init(self) -> None:

        for module in self.modules():

            if isinstance(module, networks.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, networks.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        return self