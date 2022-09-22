import cv2
import torch
from unet import UNet
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

#img = cv2.imread("data/imgs/20150919_174151_image36.png")
img = Image.open("test_image.jpg")
img = np.asarray(img).transpose((2, 0, 1))
img = torch.from_numpy(img)
img = img.unsqueeze(0)
print(img.shape)
#cv2.imshow("image",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

model = UNet(channels=3, classes=2)
model.load_state_dict(torch.load("weights/checkpoint_epoch50.pth", map_location=device))

model.to(device)
model.eval()
#img = torch.from_numpy(img)

img = img.to(device=device, dtype=torch.float32)

tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((600, 600)),
            transforms.ToTensor()
        ])

with torch.no_grad():
    output = model(img)
    probs = F.softmax(output, dim=1)[0]
    full_mask = tf(probs.cpu()).squeeze()
    mask = F.one_hot(full_mask.argmax(dim=0), model.classes).permute(2, 0, 1).numpy()
result = mask_to_image(mask)
result.save("mask.png")


#cv2.imshow("image",mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()