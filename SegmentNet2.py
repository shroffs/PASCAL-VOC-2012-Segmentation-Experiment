import torch.nn as nn
import torch

# download state dictionary of pretrained VGG16
PATH = "./"
torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/vgg16_bn-6c64b313.pth', model_dir=PATH,
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
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
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
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))

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

        self.bottleneck = bottleneck(512, 4096)  # 8x8 o

        # Expanding
        self.expand1_1 = nn.ConvTranspose2d(4096, 512, 3, dilation=2)  # 8x8 -> 12x12
        self.expand1_2 = nn.ConvTranspose2d(512, 512, 3, dilation=2)  # 12x12 -> 16x16
        self.expand2_1 = nn.ConvTranspose2d(512, 256, 3, dilation=3)  # 16x16 -> 22x22
        self.expand2_2 = nn.ConvTranspose2d(256, 256, 3, dilation=5)  # 22x2 -> 32x32
        self.bn2 = nn.BatchNorm2d(256)
        self.expand3_1 = nn.ConvTranspose2d(256 + 512, 256, 3, dilation=3)  # 32x32 w/s -> 38x38
        self.expand3_2 = nn.ConvTranspose2d(256, 128, 3, dilation=3)  # 38x38 -> 44x44
        self.expand3_3 = nn.ConvTranspose2d(128, 128, 3, dilation=4)  # 44x44 -> 52x52
        self.bn3 = nn.BatchNorm2d(128)
        self.expand3_4 = nn.ConvTranspose2d(128, 128, 3, dilation=3)  # 52x52 -> 58x58
        self.expand3_5 = nn.ConvTranspose2d(128, 128, 3, dilation=3)  # 58x58 -> 64x64
        self.expand4_1 = nn.ConvTranspose2d(128 + 256, 128, 3, dilation=1, stride=2)  # 64x64 w/s-> 129x129
        self.expand4_2 = nn.Conv2d(128, 64, 2)  # 129x129-> 128x128
        self.bn4 = nn.BatchNorm2d(64)
        self.expand5_1 = nn.ConvTranspose2d(64 + 128, 64, 3, dilation=1, stride=2)  # 128x128 w/s-> 257x257
        self.expand5_2 = nn.Conv2d(64, 64, 2)  # 257x257-> 256x256
        self.bn5 = nn.BatchNorm2d(64)
        self.expand6_1 = nn.Conv2d(64 + 64, 64, 3, padding=1)  # 256x256-> 256x256
        self.conv7 = nn.Conv2d(64, 21, 1)

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

        x = self.relu(self.expand1_1(x))
        x = self.relu(self.expand1_2(x))

        x = self.relu(self.expand2_1(x))
        x = self.bn2(self.relu(self.expand2_2(x)))

        x = torch.cat((skip4, x), dim=1)
        x = self.relu(self.expand3_1(x))
        x = self.relu(self.expand3_2(x))
        x = self.bn3(self.relu(self.expand3_3(x)))
        x = self.relu(self.expand3_4(x))
        x = self.relu(self.expand3_5(x))

        x = torch.cat((skip3, x), dim=1)
        x = self.relu(self.expand4_1(x))
        x = self.bn4(self.relu(self.expand4_2(x)))

        x = torch.cat((skip2, x), dim=1)
        x = self.relu(self.expand5_1(x))
        x = self.bn5(self.relu(self.expand5_2(x)))

        x = torch.cat((skip1, x), dim=1)
        x = self.relu(self.expand6_1(x))

        x = self.conv7(x)

        return x


if __name__ == '__main__':
    model = SegmentNet()
    model.eval()

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())