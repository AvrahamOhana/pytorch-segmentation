import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataset import ApplesDataset
from utils.diceloss import dice_loss
from utils.eval import evaluate
from torch.utils.data import DataLoader, random_split
import torch
import wandb
from unet import UNet
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.nn.functional as F



val_percent = 0.1
epochs = 10
batch_size = 1
learning_rate = 1e-4
amp = True
save_checkpoint= True


# init wandb
experiment = wandb.init(project="Apples-Segmentation", entity="avrahamiko")
experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint,
                                  amp=amp))



# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# augmentations
transform = A.Compose([
    A.Resize(720, 720),
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

# Dataloaders
train_loader = DataLoader(train_set, shuffle=True, batch_size=4, num_workers=10, pin_memory=True)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, num_workers=10, pin_memory=True)

# create the network
model = UNet(channels=3, classes=2)

# define loss and optimizer
dir_checkpoint = "weights"
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
criterion = CrossEntropyLoss()
global_step = 0

model.to(device)


# train loop
for epoch in range(1, epochs+1):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader):
        images, true_masks = batch
        images = images.to(device, dtype=torch.float32)
        true_masks = true_masks.to(device, dtype=torch.long)
        
        # forward [use auto mixed percision]
        with torch.cuda.amp.autocast(enabled=amp):
            masks_pred = model(images)
            loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, model.classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
        # backward
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        
        # log parameters
        global_step += 1
        epoch_loss += loss.item()
        
        experiment.log({
            'train loss': loss.item(),
            'step': global_step,
            'epoch': epoch
        })
        
        # Evaluation round
        division_step = (n_train // (10 * batch_size))
        if division_step > 0:
            if global_step % division_step == 0:
                

                val_score = evaluate(model, val_loader, device)
                scheduler.step(val_score)
                
                
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                })

    # save weights
    if save_checkpoint:
        torch.save(model.state_dict(), "{}/checkpoint_epoch{}.pth".format(dir_checkpoint,epoch))