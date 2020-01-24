import torch.nn as nn
import torch

#download state dictionary of pretrained VGG16
PATH = "./"
torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir=PATH,
                                   progress=True)
class contract2(nn.Module):
    """Contract Layer with 2 convolutions
    """

    def __init__(self, in_channel, out_channel):
        super(contract2, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.01)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        skip = x.clone()

        return x, skip


class contract3(nn.Module):
    """Contract Layer with 2 convolutions
    """

    def __init__(self, in_channel, out_channel):
        super(contract3, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.01)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.conv3 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)

    def forward(self, x):
        """ Residual connection between all sequential conv layers and returns skip to be use layer in expanding output same size

        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        skip = x.clone()

        return x, skip


class bottleneck(nn.Module):
    """bottleneck layer in the middle of the network
    """

    def __init__(self, in_channel, out_channel):
        super(bottleneck, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))

        return x


class expand(nn.Module):

    def __init__(self, in_channel, out_channel, has_skip=False):
        super(expand, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = nn.ReLU()
        self.has_skip = has_skip
        if self.has_skip:
            self.skip_conv = nn.Conv2d(self.in_channel, self.in_channel, 1)

        self.convt = nn.ConvTranspose2d(self.in_channel, self.out_channel, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(self.out_channel)

    def forward(self, x, skip):

        if self.has_skip:
            skip = self.skip_conv(skip)
            x = torch.add(x, skip)
        x = self.bn1(self.relu(self.convt(x)))
        x = x[:,:,:-1,:-1]
        return x


class SegmentNet2(nn.Module):

    def __init__(self):
        super(SegmentNet2, self).__init__()

        # Contracting
        self.pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()

        self.contract1 = contract2(3, 64)  # 256x256 o
        self.contract2 = contract2(64, 128)  # 128x128 o
        self.contract3 = contract3(128, 256)  # 64x64 o
        self.contract4 = contract3(256, 512)  # 32x32 o
        self.contract5 = contract3(512, 512)  # 16x16 o

        self.bottleneck1 = bottleneck(512, 4096)  # 8x8 o
        self.bottleneck2 = bottleneck(4096, 2048)

        # Expanding
        self.expand1 = expand(2048, 512, has_skip=False)  # 8x8 i
        self.expand2 = expand(512, 512, has_skip=False)  # 16x16 i
        self.expand3 = expand(512, 256, has_skip=True)  # 32x32 i
        self.expand4 = expand(256, 128, has_skip=True)  # 64x64i
        self.expand5 = expand(128, 64, has_skip=True)  # 128x128 i
        self.expand6_1 = nn.Conv2d(64, 64, 3, padding=1)  # 256x256 i
        self.bn6_1 = nn.BatchNorm2d(64)
        self.expand6_2 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 22, 1)

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

        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        x = self.expand1(x, None)
        x = self.expand2(x, None)
        x = self.expand3(x, skip4)
        x = self.expand4(x, skip3)
        x = self.expand5(x, skip2)
        x = torch.add(x, skip1)
        x = self.bn6_1(self.relu(self.expand6_1(x)))
        x = self.bn6_2(self.relu(self.expand6_2(x)))

        x = self.conv7(x)

        return x


if __name__ == '__main__':
    model = SegmentNet2()
    model.eval()