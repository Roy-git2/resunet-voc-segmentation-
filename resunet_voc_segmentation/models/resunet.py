# ResUNet model architecture
import torch.nn as nn
import timm
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, features_only=True)

    def forward(self, x):
        return self.backbone(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class ResUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution = DownSample()
        self.bottle_neck = DoubleConv(2048, 2048)
        self.up_convolution_1 = UpSample(2048, 1024)
        self.up_convolution_2 = UpSample(1024, 512)
        self.up_convolution_3 = UpSample(512, 256)
        self.up_convolution_3_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.up_convolution_4 = UpSample(128, 64)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        down1, down2, down3, down4, down5 = self.down_convolution(x)
        b = self.bottle_neck(down5)
        up1 = self.up_convolution_1(b, down4)
        up2 = self.up_convolution_2(up1, down3)
        up3 = self.up_convolution_3(up2, down2)
        up3_1 = self.up_convolution_3_1(up3)
        up4 = self.up_convolution_4(up3_1, down1)
        return self.out(up4)
