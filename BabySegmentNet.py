import torch.nn as nn
import torch


class b_res_contract(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(b_res_contract, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()

        self.skip_conv = nn.Conv2d(self.in_channel, self.out_channel, 1, bias=False)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        """ Residual connection between all sequential conv layers and returns skip to be use layer in expanding output same size

        """
        x = self.bn1(self.relu(self.conv1(x)))
        return x

class b_res_expand(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(b_res_expand, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()

        self.skip_conv = nn.Conv2d(self.in_channel, self.out_channel, 1, bias=False)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        """Residual connection between all sequential conv layers and takes in skip from contracting layer of same size

        """
        x = self.bn1(self.relu(self.conv1(x)))

        return x

class BabySegmentNet(nn.Module):

    def __init__(self):

        super(BabySegmentNet, self).__init__()

        self.norm = nn.LayerNorm((3, 512, 512), eps=1e-5)

        # Contracting
        self.pool = nn.MaxPool2d(2, 2)
        self.contract1 = b_res_contract(3, 128)
        self.contract2 = b_res_contract(128, 256)
        self.contract3 = b_res_contract(256,512)

        # Expanding
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.expand1 = b_res_expand(512, 256)
        self.expand2 = b_res_expand(256, 128)

        self.conv10 = nn.Conv2d(128, 21, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax2d()


    def forward(self, x):
        """Network predicts labels for every pixel in the arr x

        Args:
            x: 3x512x512 image

        Returns:
            x: 21x512x512 array of predictions
        """
        x = self.norm(x)

        x = self.contract1(x)
        x = self.pool(x)
        x  = self.contract2(x)
        x = self.pool(x)
        x  = self.contract3(x)

        x = self.upsample(x)
        x = self.expand1(x)
        x = self.upsample(x)
        x = self.expand2(x)

        x = self.relu(self.conv10(x))

        return x






