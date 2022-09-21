from unet.Blocks import ConvBlock, DownBlock, UpBlock
from .unet import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, channels=3, classes=2):
        super(UNet, self).__init__()
        self.channels = channels
        self.classes  = classes
        
        # 1. input layer
        self.input_conv = ConvBlock(self.channels, 64)
        
        # 2. encoder layers
        self.d1 = DownBlock(64 , 128)
        self.d2 = DownBlock(128, 256)
        self.d3 = DownBlock(256, 512)
        self.d4 = DownBlock(512, 512)
        
        
        # 3. decoder layers
        self.u1 = UpBlock(1024, 256)
        self.u1 = UpBlock(512 , 128)
        self.u1 = UpBlock(256 , 64)
        self.u1 = UpBlock(128 , 64)
        
        # 4. output layer
        self.output_conv = nn.Conv2d(64, classes, kernel_size=1)
        
    def forward(self, x):
        # 1. input layer
        x_in = self.input_conv(x)
        
        # 2. encoder
        x1 = self.d1(x_in)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        
        # 3. decoder and concat
        x = self.u1(x4, x3)
        x = self.u2(x , x2)
        x = self.u3(x , x1)
        x = self.u4(x , x_in)
        
        # 4. output layer
        x = self.output_conv(x)
        return x