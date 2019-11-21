import torch.nn as nn
import torch


class SegmentNet(nn.Module):

    def __init__(self):
        ''' Initialized layers to network
            note: padding 1 chosen to preserve shape of inputs relative to outputs
        '''

        super(SegmentNet, self).__init__()

        #Contracting
        self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, momentum=0.1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, momentum=0.1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, momentum=0.1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, momentum=0.1)

        self.conv5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(1024, momentum=0.1)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(1024, momentum=0.1)

        #Expanding
        self.conv6_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv6_2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(512, momentum=0.1)

        self.conv7_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv7_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(256, momentum=0.1)

        self.conv8_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn8_1 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv8_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn8_2 = nn.BatchNorm2d(128, momentum=0.1)

        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv9_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn9_2 = nn.BatchNorm2d(64, momentum=0.1)

        self.conv10_1 = nn.Conv2d(64, 21, 3,)
        self.bn10_1 = nn.BatchNorm2d(21)

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        """Network predicts labels for every pixel in the arr x

        Args:
            x: 3x512x512 image

        Returns:
            x: 21x512x512 array of predictions
        """
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        skip1 = x.clone()
        x = self.pool(x)

        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        skip2 = x.clone()
        x = self.pool(x)

        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        skip3 = x.clone()
        x = self.pool(x)

        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        skip4 = x.clone()
        x = self.pool(x)

        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))

        x = self.upsample(x)
        x = self.relu(self.bn6_1(self.conv6_1(x)))
        x = torch.cat([skip4, x], dim=0)
        x = self.relu(self.bn6_2(self.conv6_2(x)))

        x = self.upsample(x)
        x = self.relu(self.bn7_1(self.conv7_1(x)))
        x = torch.cat([skip3, x], dim=0)
        x = self.relu(self.bn7_2(self.conv7_2(x)))

        x = self.upsample(x)
        x = self.relu(self.bn8_1(self.conv8_1(x)))
        x = torch.cat([skip2, x], dim=0)
        x = self.relu(self.bn8_2(self.conv8_2(x)))

        x = self.upsample(x)
        x = self.relu(self.bn9_1(self.conv9_1(x)))
        x = torch.cat([skip1, x], dim=0)
        x = self.relu(self.bn9_2(self.conv9_2(x)))

        x = self.relu(self.bn10_1(self.conv10_1(x)))
        x = self.softmax(x)

        return x



