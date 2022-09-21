import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataset import ApplesDataset
from torch.utils.data import DataLoader, random_split
import torch
import wandb

val_percent = 0.1
epochs = 10
batch_size = 4
learning_rate = 1e-5
amp = True
save_checkpoint= True

transform = A.Compose([
    A.Resize(416, 416),
    A.RandomBrightnessContrast(p=0.4),
    A.HorizontalFlip(p=0.5),
    ToTensorV2(),
])

# create dataset
dataset = ApplesDataset('data/imgs', 'data/masks', transform=transform)


# split into train / validation 
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# init wandb
experiment = wandb.init(project="Apples-Segmentation", entity="avrahamiko")
experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))