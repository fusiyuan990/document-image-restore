import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return x + res

class SRResNet(nn.Module):
    def __init__(self, scale=4, num_res_blocks=8):
        super().__init__()
        self.conv_in = nn.Sequential(nn.Conv2d(1,64,9,padding=4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.conv_mid = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64))
        self.upsample = nn.Sequential(
            nn.Conv2d(64,64*scale*scale,3,padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU()
        )
        # PixelShuffle后通道回到64，conv_out输入通道改为64
        self.conv_out = nn.Conv2d(64,1,9,padding=4)
    def forward(self, x):
        x1 = self.conv_in(x)
        res = self.res_blocks(x1)
        x2 = self.conv_mid(res)
        x = x1 + x2
        x = self.upsample(x)
        out = self.conv_out(x)
        return out