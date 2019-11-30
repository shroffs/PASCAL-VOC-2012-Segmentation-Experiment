import cv2
import matplotlib.pyplot as plt
import numpy as np
from SegmentNet import SegmentNet
import torch
import scipy.special as scsp
import sys

np.set_printoptions(threshold=sys.maxsize)

#set device to GPU
device = torch.device(0)

#Load in net and parameters
net_state_dict = ".\\TrainedNetT1"
net = SegmentNet().type(torch.cuda.FloatTensor)
net.load_state_dict(torch.load(net_state_dict))
net.eval()

img_path = "C:\\Users\\Spencer\\Documents\\GitHub\\PASCAL-VOC-2012-Segmentation-Experiement\\VOCdevkit\\VOC2012\\JPEGImages\\2007_002370.jpg"
img = cv2.imread(img_path, 1)


def encode_image(img): #any image HxWx3
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
    # swap axes for CxHxW array
    img = np.array(img)
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 1)
    img = torch.tensor(img).type(torch.cuda.FloatTensor)
    img = img.unsqueeze_(0)
    return img

def decode_image(label): #1x512x512
    label = label.to('cpu').detach().numpy()
    label = np.squeeze(label, 0)
    label = np.swapaxes(label, 0, 1)
    label = np.swapaxes(label, 1, 2)
    h,w,c = label.shape
    res = np.zeros((h, w, 3), dtype=int)
    classes = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                    [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64],
                    [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192],
                    [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0],
                    [0, 192, 128], [128, 64, 0]]

    for i in range(len(label)):
        for j in range(len(label[i])):
            res[i][j] = classes[np.argmax(label[i][j])]
    return res



cv2.imshow('img', img)
cv2.waitKey(0)
img_enc = encode_image(img)
label = net(img_enc)
res = decode_image(label)
plt.imshow(res)
plt.show()


