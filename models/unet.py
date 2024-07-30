import torch # type: ignore 
from torch import nn # type: ignore
from torch.nn import Sequential, Upsample, ConvTranspose2d, MaxPool2d, Module, Conv2d, BatchNorm2d, ReLU # type: ignore 
import torch.nn.functional as F # type: ignore

class DoubleConv(Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = Sequential(
            Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            BatchNorm2d(mid_channels),
            ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv_maxpool = Sequential(
            MaxPool2d(2),
            DoubleConv(in_channels, out_channels) # (I put maxpool before, but in the model itself I use doubleconv first in UNet, so the sequence of modules is correct)
        )

    def forward(self, x):
        return self.conv_maxpool(x)

class Up(Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(Module):
    def __init__(self, img_channels, n0, num_classes, bilinear=True):
        super(UNet, self).__init__()
        self.stages = [img_channels] + [n0 * 2**i for i in range(6)]
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        assert self.img_channels > 0, "img_channels must be greater than 0"
        assert self.num_classes > 0, "num_classes must be greater than 0"

        self.inc = DoubleConv(img_channels, self.stages[1])
        self.down1 = Down(self.stages[1], self.stages[2])
        self.down2 = Down(self.stages[2], self.stages[3])
        self.down3 = Down(self.stages[3], self.stages[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(self.stages[4], self.stages[5] // factor)

        self.up1 = Up(self.stages[5], self.stages[4] // factor, bilinear)
        self.up2 = Up(self.stages[4], self.stages[3] // factor, bilinear)
        self.up3 = Up(self.stages[3], self.stages[2] // factor, bilinear)
        self.up4 = Up(self.stages[2], self.stages[1], bilinear)
        self.outc = OutConv(n0, num_classes)
        
    def forward(self, x):
        xinc = self.inc(x)
        x1 = self.down1(xinc)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, xinc)
        return self.outc(x)