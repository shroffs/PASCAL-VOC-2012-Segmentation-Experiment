from SegmentNet2 import SegmentNet2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.special as scisp
import sys
import os
import random

#set device to GPU
device = torch.device(0)

#Load in net and parameters
net_state_dict = "./TrainedNet1"
net = SegmentNet2().type(torch.cuda.FloatTensor)
net.load_state_dict(torch.load(net_state_dict))
net.eval()


def random_crop(img, size):
    """Takes HxWxC img and returns SizexSizexC img
    """
    h, w = img.shape[0], img.shape[1]

    # pick a random valid point to be the upper left corner of the crop
    p1 = random.randint(0, h - size)
    p2 = random.randint(0, w - size)

    # return the cropped img
    return img[p1:p1 + size, p2:p2 + size, :]

def encode_image(img): #any image HxWx3
    #crop
    img = random_crop(img, 256)
    # swap axes for CxHxW array
    img = np.array(img)
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 1)

    img = torch.tensor(img).type(torch.cuda.FloatTensor)
    #make 1xCxHxW
    img = img.unsqueeze_(0)
    return img

def decode_image(label): #1x512x512

    label = label.to('cpu').detach()

    #make HxWxC
    label = torch.squeeze(label, 0)
    label = torch.argmax(label, dim=0).numpy()

    h,w = label.shape

    #initalize output image
    res = np.zeros((h, w, 3), dtype=int)

    #create class pixel values
    classes = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                    [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64],
                    [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192],
                    [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0],
                    [0, 192, 128], [128, 64, 0]]
    
    for i in range(len(label)):
        for j in range(len(label[i])):
            res[i][j] = classes[label[i][j]]

    return res


if __name__=='__main__':

    #choose image to segment
    dir_path = './VOCdevkit/VOC2012/JPEGImages'
    filenames = os.listdir(dir_path)
    idx = int(sys.argv[1])
    img_path = os.path.join(dir_path, filenames[idx])

    #Put image through network
    img = cv2.imread(img_path)
    img_enc = encode_image(img)
    label = net(img_enc)
    res = decode_image(label)

    #Show image and output
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(img)
    f.add_subplot(1,2,2)
    plt.imshow(res)
    plt.show()


