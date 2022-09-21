from unet.Blocks import ConvBlock, DownBlock, UpBlock
from .unet import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, channels, classes):
        super(UNet, self).__init__()
        self.channels = channels
        self.classes  = classes
        
        # 1. input layer
        self.input_conv = ConvBlock(channels, 64)
        
        # 2. encoder layers
        self.d1 = DownBlock(64 , 128)
        self.d2 = DownBlock(128, 256)
        self.d3 = DownBlock(256, 512)
        self.d4 = DownBlock(512, 1024)
        
        
        # 3. decoder layers
        self.u1 = UpBlock(1024, 512)
        self.u2 = UpBlock(512 , 256)
        self.u3 = UpBlock(256 , 128)
        self.u4 = UpBlock(128 , 64)
        
        # 4. output layer
        self.output_conv = nn.Conv2d(64, classes, kernel_size=1)
        
    def forward(self, x):
        # 1. input layer
        x1 = self.input_conv(x)
        
        # 2. encoder
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        
        # 3. decoder and concat
        x = self.u1(x5, x4)
        x = self.u2(x , x3)
        x = self.u3(x , x2)
        x = self.u4(x , x1)
        
        # 4. output layer
        out = self.output_conv(x)
        return out