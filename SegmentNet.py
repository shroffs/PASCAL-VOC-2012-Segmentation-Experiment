import torch.nn as nn
import torch

class res_contract(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(res_contract, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout2d(0.1)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        """ Residual connection between all sequential conv layers and returns skip to be use layer in expanding output same size

        """
        x = self.bn1(self.tanh(self.conv1(x)))
        x = self.bn2(self.tanh(self.conv2(x)))
        skip = x.clone()
        x = self.dropout(x)

        return x, skip


class res_expand(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(res_expand, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout2d(0.1)

        self.conv1 = nn.Conv2d(self.in_channel + self.out_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x, skip):
        """Residual connection between all sequential conv layers and takes in skip from contracting layer of same size

        """
        x = torch.cat((skip, x), dim=1)
        x = self.bn1(self.tanh(self.conv1(x)))
        x = self.bn2(self.tanh(self.conv2(x)))
        x = self.dropout(x)

        return x


class SegmentNet(nn.Module):

    def __init__(self):
        super(SegmentNet, self).__init__()

        # Contracting
        self.pool = nn.MaxPool2d((2, 2))

        self.contract1 = res_contract(3, 64)
        self.contract2 = res_contract(64, 128)
        self.contract3 = res_contract(128, 256)
        self.contract4 = res_contract(256, 512)

        self.contract5 = res_contract(512, 1024)

        # Expanding
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.expand1 = res_expand(1024, 512)
        self.expand2 = res_expand(512, 256)
        self.expand3 = res_expand(256, 128)
        self.expand4 = res_expand(128, 64)

        self.conv10 = nn.Conv2d(64, 21, 1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        """Network predicts labels for every pixel in the arr x

        Args:
            x: 3xHxW image

        Returns:
            x: 1xHxW array of predictions
        """

        x, skip1 = self.contract1(x)
        x = self.pool(x)
        x, skip2 = self.contract2(x)
        x = self.pool(x)
        x, skip3 = self.contract3(x)
        x = self.pool(x)
        x, skip4 = self.contract4(x)
        x = self.pool(x)
        x, _ = self.contract5(x)

        x = self.upsample(x)
        x = self.expand1(x, skip4)
        x = self.upsample(x)
        x = self.expand2(x, skip3)
        x = self.upsample(x)
        x = self.expand3(x, skip2)
        x = self.upsample(x)
        x = self.expand4(x, skip1)

        x = self.conv10(x)
        x = self.softmax(x)

        return x