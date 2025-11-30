import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import open3d

class PCGen(nn.Module):
    def __init__(self):
        super().__init__()

        # input 3 x 192 x 256 image
        self.inputconv16 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # input 16 1 x 192 x 256 feature maps
        self.conv16 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        # input 16 1 x 192 x 256 feature maps
        self.stridedconv32 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        # input 32 1 x 96 x 128 feature maps
        self.conv32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        # input 32 1 x 96 x 128 feature maps
        self.stridedconv64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        # input 64 1 x 48 x 64 feature maps
        self.conv64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # input 64 1 x 48 x 64 feature maps
        self.stridedconv128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        # input 128 1 x 24 x 32 feature maps
        self.conv128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # input 128 1 x 24 x 32 feature maps
        self.stridedconv256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        # input 256 1 x 12 x 16 feature maps
        self.conv256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        # input 256 1 x 12 x 16 feature maps
        self.stridedconv512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        # input 512 1 x 6 x 8 feature maps
        self.conv512 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # need to figure out that last conv layer

        # pred fc
        # input 512 1 x 6 x 8 feature 
        self.predfc = nn.Linear(in_features=32, out_features=32)

        layers = []
        layers.append(self.inputconv16)
        layers.append(nn.ReLU())
        layers.append(self.conv16)
        layers.append(nn.ReLU())
        layers.append(self.stridedconv32)
        layers.append(nn.ReLU())
        layers.append(self.conv32)
        layers.append(nn.ReLU())
        layers.append(self.conv32)
        layers.append(nn.ReLU())

        print("model gen success!")




if __name__ == "__main__":
    device = 'cpu'
    if torch.accelerator.is_available():
        device =torch.accelerator

    print(f"Using {device} device")
    pcgen = PCGen()