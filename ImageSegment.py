import cv2
import numpy as np
from SegmentNet import SegmentNet
import torch

#set device to GPU
device = torch.set_device(0)

#Load in net and parameters
net_state_dict =
net = SegmentNet().to(device)
net.load_state_dict(torch.load(net_state_dict))
net.eval()

img_path =
img = cv2.imread(img_path)
h, w, c = img.shape

def encode_image(img): #any image HxWxC
    img = cv2.resize(img, (512,512))
    img = np.swapaxes(0,2)
    return img

def decode_image(label): #21x512x512
    img = np.zeros (512,512,3)

    label = np.swapaxes(label, 0,2)

    classes = [[192, 224, 224], [0, 0, 0], [0, 0, 128],
               [0, 128, 0], [0, 128, 128], [128, 0, 0],
               [128, 0, 128], [128, 128, 0], [128, 128, 128],
               [128, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
               [128, 0, 64], [128, 0, 192], [128, 128, 192],
               [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128],
               [128, 64, 0]]

    keys = range(21)

    dictionary = dict(zip(keys, classes))

    for i in range(len(img)):
        for j in range(len(img[i])):
            img[i][j] = dictionary[label[i][j]]




