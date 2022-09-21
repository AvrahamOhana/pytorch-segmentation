from torch.utils.data import Dataset
import cv2
import os
class ApplesDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ids = [file.split()[0] for file in os.listdir(images_dir) if not file.startswith('.')]
        self.transform = transform
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        name = self.ids[idx]
        image = cv2.imread("{}/{}".format(self.images_dir, name))
        mask = cv2.imread("{}/{}".format(self.masks_dir, name))
        
    
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            transformed_mask = transformed_mask[:,:,-1]
            transformed_mask[transformed_mask > 0] = 1
        return transformed_image, transformed_mask