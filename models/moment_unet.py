import os
import sys

import torch
from torch import nn
from torch.nn import Sequential, Module
import torch.nn.functional as F

path = os.getcwd()
par_path = os.path.abspath(os.pardir)
sys.path.append(par_path)

from moment_kernels import *

class DoubleConv(Module):
    def __init__(self, in_scalars, in_vectors, out_scalars, out_vectors, mid_scalars=None, mid_vectors=None):
        super(DoubleConv, self).__init__()
        if not mid_scalars:
            mid_scalars = out_scalars
        if not mid_vectors:
            mid_vectors = out_vectors
        self.double_conv = Sequential(
            ScalarVectorToScalarVector(in_scalars, in_vectors, mid_scalars, mid_vectors, kernel_size=3, padding=1),
            ScalarVectorBatchnorm(mid_scalars, mid_vectors),
            ScalarVectorSigmoid(mid_scalars),
            ScalarVectorToScalarVector(mid_scalars, mid_vectors, out_scalars, out_vectors, kernel_size=3, padding=1),
            ScalarVectorBatchnorm(out_scalars, mid_vectors),
            ScalarVectorSigmoid(out_scalars)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(Module):
    def __init__(self, in_scalars, in_vectors, out_scalars, out_vectors):
        super(Down, self).__init__()
        self.conv_maxpool = Sequential(
            Downsample(),
            DoubleConv(in_scalars, in_vectors, out_scalars, out_vectors)
        )

    def forward(self, x):
        return self.conv_maxpool(x)

class Up(Module):
    def __init__(self, in_scalars, in_vectors, skip_scalars, skip_vectors, out_scalars, out_vectors):
        super(Up, self).__init__()
        self.up = Upsample()
        self.conv = DoubleConv(in_scalars + skip_scalars, in_vectors + skip_vectors, out_scalars, out_vectors)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(Module):
    def __init__(self, in_scalars, in_vectors, out_scalars, out_vectors):
        super(OutConv, self).__init__()
        self.conv = ScalarVectorToScalarVector(in_scalars, in_vectors, out_scalars, out_vectors, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MomentUNet(Module):
    def __init__(self, img_channels, n0, num_classes):
        super(MomentUNet, self).__init__()
        self.stages = [img_channels] + [n0 * 2**i for i in range(6)]
        self.img_channels = img_channels
        self.num_classes = num_classes

        assert self.img_channels > 0, "img_channels must be greater than 0"
        assert self.num_classes > 0, "num_classes must be greater than 0"

        self.inc = DoubleConv(img_channels, 0, self.stages[1] // 2, self.stages[1] // 2)
        self.down1 = Down(self.stages[1] // 2, self.stages[1] // 2, self.stages[2] // 2, self.stages[2] // 2)
        self.down2 = Down(self.stages[2] // 2, self.stages[2] // 2, self.stages[3] // 2, self.stages[3] // 2)
        self.down3 = Down(self.stages[3] // 2, self.stages[3] // 2, self.stages[4] // 2, self.stages[4] // 2)
        self.down4 = Down(self.stages[4] // 2, self.stages[4] // 2, self.stages[5] // 2, self.stages[5] // 2)

        self.up1 = Up(self.stages[5] // 2, self.stages[5] // 2, self.stages[4] // 2, self.stages[4] // 2, self.stages[4] // 2, self.stages[4] // 2)
        self.up2 = Up(self.stages[4] // 2, self.stages[4] // 2, self.stages[3] // 2, self.stages[3] // 2, self.stages[3] // 2, self.stages[3] // 2)
        self.up3 = Up(self.stages[3] // 2, self.stages[3] // 2, self.stages[2] // 2, self.stages[2] // 2, self.stages[2] // 2, self.stages[2] // 2)
        self.up4 = Up(self.stages[2] // 2, self.stages[2] // 2, self.stages[1] // 2, self.stages[1] // 2, self.stages[1] // 2, self.stages[1] // 2)
        self.outc = ScalarVectorToScalarVector(n0 + n0 // 2, n0 + n0 // 2, num_classes, 0, kernel_size=1)

    def forward(self, x):
        xinc = self.inc(x)
        print(f"xinc: {xinc.size()}")
        x1 = self.down1(xinc)
        print(f"x1: {x1.size()}")
        x2 = self.down2(x1)
        print(f"x2: {x2.size()}")
        x3 = self.down3(x2)
        print(f"x3: {x3.size()}")
        x4 = self.down4(x3)
        print(f"x4: {x4.size()}")

        x = self.up1(x4, x3)
        print(f"x after up1: {x.size()}")
        x = self.up2(x, x2)
        print(f"x after up2: {x.size()}")
        x = self.up3(x, x1)
        print(f"x after up3: {x.size()}")
        x = self.up4(x, xinc)
        print(f"x after up4: {x.size()}")
        return self.outc(x)