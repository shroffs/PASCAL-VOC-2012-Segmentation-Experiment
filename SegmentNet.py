import torch.nn as nn
import torch
import torchvision.models as models

PATH = "./"
torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/vgg16_bn-6c64b313.pth', model_dir=PATH, progress=True)

class contract2(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(contract2, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        """ Residual connection between all sequential conv layers and returns skip to be use layer in expanding output same size

        """
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        skip = x.clone()
        x = self.dropout(x)

        return x, skip

class contract3(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(contract3, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv3 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        """ Residual connection between all sequential conv layers and returns skip to be use layer in expanding output same size

        """
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        skip = x.clone()
        x = self.dropout(x)

        return x, skip

class bottleneck(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(bottleneck, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)


    def forward(self, x):
        """ Residual connection between all sequential conv layers and returns skip to be use layer in expanding output same size

        """
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.dropout(x)

        return x


class expand(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(expand, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)

        self.conv1 = nn.Conv2d(self.in_channel + self.out_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x, skip=None, no_skip = False):
        """Residual connection between all sequential conv layers and takes in skip from contracting layer of same size

        """
        if not no_skip:
            x = torch.cat((skip, x), dim=1)
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.dropout(x)

        return x


class SegmentNet(nn.Module):

    def __init__(self):
        super(SegmentNet, self).__init__()

        # Contracting
        self.pool = nn.MaxPool2d((2, 2))

        self.contract1 = contract2(3, 64)
        self.contract2 = contract2(64, 128)
        self.contract3 = contract3(128, 256)
        self.contract4 = contract3(256, 512)
        self.contract5 = contract3(512, 512)

        self.bottleneck = bottleneck(512, 4096)

        # Expanding
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.expand1 = expand(4096, 512)
        self.expand2 = expand(512, 512)
        self.expand3 = expand(512, 256)
        self.expand4 = expand(256, 128)
        self.expand5 = expand(128, 64)

        self.conv10 = nn.Conv2d(64, 21, 1)

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
        x = self.pool(x)

        x = self.bottleneck(x)

        x = self.upsample(x)
        x = self.expand1(x, no_skip=True)
        x = self.upsample(x)
        x = self.expand2(x, skip4)
        x = self.upsample(x)
        x = self.expand3(x, skip3)
        x = self.upsample(x)
        x = self.expand4(x, skip2)
        x = self.upsample(x)
        x = self.expand5(x, skip1)

        x = self.conv10(x)

        return x

"""
model = SegmentNet()
model.load_state_dict(torch.load(PATH), strict=False)
model.eval()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
"""