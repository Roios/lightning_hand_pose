import torch
import torch.nn as nn
import pytorch_lightning as pl


class ConvolutionBlock(pl.LightningModule):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_depth),
            nn.Conv2d(in_channels=in_depth, out_channels=out_depth, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_depth),
            nn.Conv2d(in_channels=out_depth, out_channels=out_depth, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(pl.LightningModule):
    def __init__(self, in_channel=3, out_channel=21, neurons=16):
        super().__init__()

        self.conv_down1 = ConvolutionBlock(in_channel, neurons)
        self.conv_down2 = ConvolutionBlock(neurons, neurons * 2)
        self.conv_down3 = ConvolutionBlock(neurons * 2, neurons * 4)
        self.conv_bottleneck = ConvolutionBlock(neurons * 4, neurons * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv_up1 = ConvolutionBlock(neurons * 8 + neurons * 4, neurons * 4)
        self.conv_up2 = ConvolutionBlock(neurons * 4 + neurons * 2, neurons * 2)
        self.conv_up3 = ConvolutionBlock(neurons * 2 + neurons, neurons)

        self.conv_out = nn.Sequential(nn.Conv2d(neurons, out_channel, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                      nn.Sigmoid())

    def forward(self, x):
        conv_d1 = self.conv_down1(x)
        conv_d2 = self.conv_down2(self.maxpool(conv_d1))
        conv_d3 = self.conv_down3(self.maxpool(conv_d2))
        conv_b = self.conv_bottleneck(self.maxpool(conv_d3))

        conv_u1 = self.conv_up1(torch.cat([self.upsample(conv_b), conv_d3], dim=1))
        conv_u2 = self.conv_up2(torch.cat([self.upsample(conv_u1), conv_d2], dim=1))
        conv_u3 = self.conv_up3(torch.cat([self.upsample(conv_u2), conv_d1], dim=1))

        out = self.conv_out(conv_u3)
        return out
