import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex, Dice
from U_net import UNet
from get_data import PolypsSegmentationDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
val_images = 0.2
dataset = PolypsSegmentationDataset()
n_val_images = round(val_images * len(dataset))
train_dataset, val_dataset = random_split(dataset, [len(dataset) - n_val_images, n_val_images])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=14, prefetch_factor=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=14, prefetch_factor=1)

class DiceLossWithLogits(nn.Module):
    def __init__(self, smooth):
        super().__init__()
        self.smooth = smooth

    def forward(self, x,y):
        x = F.sigmoid(x)
        x = x.view(-1)
        y = y.view(-1)
        intersection = x * y
        dice_coefficient = (2 * intersection.sum() + self.smooth)/(x.sum() + y.sum() + self.smooth)
        loss = 1 - dice_coefficient
        return loss

model = UNet(input_channels=3)
model = model.to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
criterion = DiceLossWithLogits(1e-7)
criterion2 = nn.BCEWithLogitsLoss()

n_epochs = 30
jaccard = JaccardIndex(task='binary',num_classes=1)
jaccard = jaccard.to(device)

if __name__ == '__main__':
    for epoch in range(n_epochs):
        loop = tqdm.tqdm(train_loader)
        for idx,(images, masks) in enumerate(loop):
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.float()
            pred = model(images)
            loss = criterion(pred, masks) + criterion2(pred,masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            masks[masks < 0.5] = 0
            masks[masks >= 0.5] = 1
            jaccard.update(pred,masks)

            loop.set_description(f'Epoch {epoch+1}/{n_epochs}')
            loop.set_postfix(loss=loss.item(), jaccard = jaccard.compute())
    jaccard.reset()