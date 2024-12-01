# --------------------------------------------------------------------------------------------------
# ------------------------------ Dataset Image Transformer and Loader ------------------------------
# --------------------------------------------------------------------------------------------------
import torchvision.transforms as transforms
import torch.utils.data as datatools

import pathlib
import random
import typing
import torch

from PIL import Image


# --------------------------------------------------------------------------------------------------
# ------------------------------------ CLASS :: Dataset Manager ------------------------------------
# --------------------------------------------------------------------------------------------------
class Dataset(datatools.Dataset):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        super(Dataset, self).__init__()

        # ----------------------------------------------------------------------------------
        # -------------------------------- Load Image Paths --------------------------------
        # ----------------------------------------------------------------------------------

        self.images = []

        if torch.cuda.is_available():
            root = pathlib.Path().resolve()
            root = root.parent
            root = root / 'drive'
            root = root / 'MyDrive'
            root = root / 'MUNIT'

        else:
            root = pathlib.Path.home() / 'Desktop/MUNIT/'

        self.images.extend((root / 'Pantheon').rglob('*.png'))
        self.images.extend((root / 'Hilda').rglob('*.png'))


        # ----------------------------------------------------------------------------------
        # ---------------------- Transform :: Randomly Sized Cropping ----------------------
        # ----------------------------------------------------------------------------------
        def randomCropResize(image: Image.Image) -> Image.Image:

            image = transforms.RandomCrop(random.randint(512, 1024))(image)
            image = image.resize((512, 512), Image.BILINEAR)

            return image


        # ----------------------------------------------------------------------------------
        # ----------------------- Transform :: Inject Gaussian Noise -----------------------
        # ----------------------------------------------------------------------------------
        def addGaussianNoise(image: torch.Tensor, mean=0.0, std=0.01) -> torch.Tensor:

            noise = torch.randn(image.size()) * std + mean
            image = image + noise
            image = torch.clamp(image, 0.0, 1.0)

            return image


        # ----------------------------------------------------------------------------------
        # --------------------------------- Set Transforms ---------------------------------
        # ----------------------------------------------------------------------------------
        self.transforms = transforms.Compose([
            randomCropResize,
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            addGaussianNoise,
        ])


    # ------------------------------------------------------------------------------------------
    # ----------------------------------- OPERATOR :: Length -----------------------------------
    # ------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.images)


    # ------------------------------------------------------------------------------------------
    # ---------------------------------- OPERATOR :: Get Item ----------------------------------
    # ------------------------------------------------------------------------------------------
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transforms(Image.open(self.images[index]))
