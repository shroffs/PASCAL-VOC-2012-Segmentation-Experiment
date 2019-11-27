import torch.nn as nn
import torch


class b_res_contract(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(b_res_contract, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.10)

        self.skip_conv = nn.Conv2d(self.in_channel, self.out_channel, 1, bias=False)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        """ Residual connection between all sequential conv layers and returns skip to be use layer in expanding output same size

        """
        a = self.skip_conv(x)
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.dropout(x)

        return x

class b_res_expand(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(b_res_expand, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.10)

        self.skip_conv = nn.Conv2d(self.in_channel, self.out_channel, 1, bias=False)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        """Residual connection between all sequential conv layers and takes in skip from contracting layer of same size

        """
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.dropout(x)


        return x

class BabySegmentNet(nn.Module):

    def __init__(self):

        super(BabySegmentNet, self).__init__()

        self.norm = nn.LayerNorm((3, 512, 512), eps=1e-5)

        # Contracting
        self.pool = nn.MaxPool2d(2, 2)
        self.contract1 = b_res_contract(3, 64)
        self.contract2 = b_res_contract(64, 128)
        self.contract3 = b_res_contract(128,512)
        self.contract4 = b_res_contract(512, 1096)

        # Expanding
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.expand1 = b_res_expand(1096, 512)
        self.expand2 = b_res_expand(512, 128)
        self.expand3 = b_res_expand(128, 64)
        self.expand4 = b_res_expand(64, 32)

        self.conv10 = nn.Conv2d(32, 21, 1)
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
        x = self.contract2(x)
        x = self.pool(x)
        x = self.contract3(x)
        x = self.pool(x)
        x = self.contract4(x)

        x = self.expand1(x)
        x = self.upsample(x)
        x = self.expand2(x)
        x = self.upsample(x)
        x = self.expand3(x)
        x = self.upsample(x)
        x = self.expand4(x)

        x = self.relu(self.conv10(x))

        return x






