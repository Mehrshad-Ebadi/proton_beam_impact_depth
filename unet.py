import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()

        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.u1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        c1 = self.d1(x)
        p1 = self.p1(c1)

        c2 = self.d2(p1)
        p2 = self.p2(c2)

        c3 = self.d3(p2)
        p3 = self.p3(c3)

        bn = self.bottleneck(p3)

        up3 = self.u3(bn)
        up3 = torch.cat([up3, c3], dim=1)
        c3d = self.dec3(up3)

        up2 = self.u2(c3d)
        up2 = torch.cat([up2, c2], dim=1)
        c2d = self.dec2(up2)

        up1 = self.u1(c2d)
        up1 = torch.cat([up1, c1], dim=1)
        c1d = self.dec1(up1)

        return self.out(c1d)