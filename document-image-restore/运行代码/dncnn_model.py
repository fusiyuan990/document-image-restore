import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=17):
        """
        DnCNN去噪网络
        :param channels: 输入图像通道数，灰度图=1，彩色图=3
        :param num_layers: 网络层数，默认17层
        """
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []

        # 第一层：卷积 + ReLU
        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # 中间层：卷积 + 批归一化 + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层：卷积输出噪声
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # 网络输出的是预测的噪声
        out = self.dncnn(x)
        return out