# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Autoencoder Main ----------------------------------------
# --------------------------------------------------------------------------------------------------
import os; os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

from Modules.Adversary import Adversary
from Modules.Encoder   import Encoder
from Modules.Decoder   import Decoder
from Dataset.Sampler   import Sampler
from Dataset.Dataset   import Dataset

import torchvision.transforms as transforms
import torch.utils.data as datatools
import torch.nn.functional as functional
import torch

from PIL import Image


# --------------------------------------------------------------------------------------------------
# ------------------------------------------- Save Image -------------------------------------------
# --------------------------------------------------------------------------------------------------
def save(eid: int, bid: int, source: torch.Tensor, target: torch.Tensor) -> None:

    output = Image.new('RGB', (1024, 512))

    source = source.cpu()
    target = target.cpu()

    output.paste(transforms.ToPILImage()(source[1]), (0,   0))
    output.paste(transforms.ToPILImage()(target[1]), (512, 0))

    output.save(f"./Output/{eid}-{bid}.png")


# --------------------------------------------------------------------------------------------------
# ---------------------------------- Dataset and Hyper-Parameters ----------------------------------
# --------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

BATCH_SIZE = 8
EPOCHS = 1

# Dataset class is responsible for loading images from indices and transforming accordingly
# Sampler class simply shuffles indices, but is maintained in case novel behaviour is needed later
dataset = datatools.DataLoader(dataset=Dataset(), sampler=Sampler(), batch_size=BATCH_SIZE)

adversary = Adversary().to(device).train()
encoder = Encoder().to(device).train()
decoder = Decoder().to(device).train()

try:
    adversary.load()
except FileNotFoundError:
    adversary.init()
    adversary.save()
    
try:
    encoder.load()
except FileNotFoundError:
    encoder.init()
    encoder.save()
    
try:
    decoder.load()
except FileNotFoundError:
    decoder.init()
    decoder.save()

a_optimizer = torch.optim.Adam(adversary.parameters(), lr=1e-5)
e_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-5)
d_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-5)


# --------------------------------------------------------------------------------------------------
# --------------------------------------- Main Training Loop ---------------------------------------
# --------------------------------------------------------------------------------------------------
labels = torch.cat((torch.ones(BATCH_SIZE, 1), torch.zeros(BATCH_SIZE, 1))).to(device)

for eid in range(EPOCHS):
    for bid, source in enumerate(dataset):

        source = source.to(device)

        a_optimizer.zero_grad()
        e_optimizer.zero_grad()
        d_optimizer.zero_grad()

        target = encoder(source)
        target = decoder(target)

        recreate = decoder.loss(source, target)

        images = torch.cat((source.clone().detach(), target.clone().detach()))
        sample = torch.randperm(images.size(0))
        images = images[sample]
        labels = labels[sample]

        predictions = adversary(images)
        positive, negative = adversary.loss(predictions, labels)

        if bid % 5 == 0:
            positive.backward()
            a_optimizer.step()

        else:
            (recreate.backward())
            e_optimizer.step()
            d_optimizer.step()

        print(f"[{eid:03}:{bid:03}] Positive: {positive.item()}")
        print(f"[{eid:03}:{bid:03}] Negative: {negative.item()}")
        print(f"[{eid:03}:{bid:03}] Recreate: {recreate.item()}")
        print(f"-----------------------------------------------")

        if not bid % 10:
            save(eid, bid, source, target)

    adversary.save()
    encoder.save()
    decoder.save()

    break