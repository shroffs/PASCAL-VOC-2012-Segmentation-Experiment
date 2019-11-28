import torch.nn as nn
import torch
import torchvision


class res_contract(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(res_contract, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout2d(0.1)

        self.skip_conv = nn.Conv2d(self.in_channel, self.out_channel, 1, bias=False)

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x):
        """ Residual connection between all sequential conv layers and returns skip to be use layer in expanding output same size

        """
        r1 = self.skip_conv(x)
        x = self.bn1(self.tanh(self.conv1(x)))
        x = torch.add(x, r1)
        r2 = x.clone()
        x = self.dropout(x)
        x = self.bn2(self.tanh(self.conv2(x)))
        x = torch.add(x, r2)
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

        self.skip_conv = nn.Conv2d(self.in_channel + self.out_channel, self.out_channel, 1, bias=False)

        self.conv1 = nn.Conv2d(self.in_channel + self.out_channel, self.out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel, eps=1e-5)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel, eps=1e-5)

    def forward(self, x, skip):
        """Residual connection between all sequential conv layers and takes in skip from contracting layer of same size

        """
        x = torch.cat((skip, x), dim=1)
        r1 = self.skip_conv(x)
        x = self.bn1(self.tanh(self.conv1(x)))
        x = torch.add(x, r1)
        x = self.dropout(x)
        r2 = x.clone()
        x = self.bn2(self.tanh(self.conv2(x)))
        x = torch.add(x, r2)
        x = self.dropout(x)

        return x

class SegmentNet(nn.Module):

    def __init__(self):

        super(SegmentNet, self).__init__()

        self.norm = nn.InstanceNorm2d(3, eps=1e-5)

        # Contracting
        self.pool = nn.MaxPool2d((2, 2), ceil_mode=True)
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
        self.expand4 = res_expand(128,64)

        self.conv10 = nn.Conv2d(64, 21, 1)


    def forward(self, x):
        """Network predicts labels for every pixel in the arr x

        Args:
            x: 3xHxW image

        Returns:
            x: 1xHxW array of predictions
        """

        x = self.norm(x)             # 3x512x512 -> 3x512x512

        x, skip1 = self.contract1(x) # 3x512x512 -> 64x512x512 (1)
        x = self.pool(x)             # 64x512x512 -> 64x256x256
        x, skip2 = self.contract2(x) # 64x256x256 -> 128x256x256 (2)
        x = self.pool(x)             # 128x1256x256 -> 128x128x128
        x, skip3 = self.contract3(x) # 128x128x128 -> 256x128x128 (3)
        x = self.pool(x)             # 256x128x128 -> 256x64x64
        x, skip4 = self.contract4(x) # 256x64x64 -> 512x64x64 (4)
        x = self.pool(x)             # 512x64x64 -> 512x32x32
        x, _ = self.contract5(x)     # 512x32x32 -> 1024x32x32

        x = self.upsample(x)         # 1024x32x32 ->1024x64x64
        x = self.expand1(x, skip4)   # 1024x64x64 and 512x64x64(4) -> 512x64x64
        x = self.upsample(x)         # 512x64x64 -> 512x128x128
        x = self.expand2(x, skip3)   # 512x128x128 and 256x128x128(3) -> 256x128x128
        x = self.upsample(x)         # 256x128x128 -> 256x256x256
        x = self.expand3(x, skip2)   # 256x256x256 and 128x256x256(2) -> 128x256x256
        x = self.upsample(x)         # 128x256x256 -> 128x512x512
        x = self.expand4(x, skip1)   # 128x512x512 and 64x512x512(1) -> 64x512x512
        x = self.conv10(x)           # 64x512x512 -> 21x512x512

        return x