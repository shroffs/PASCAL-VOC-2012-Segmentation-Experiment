import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import torch.tensor as tensor

#Dirs created by DataSeparate.py
img_path_train = "./VOCdevkit/VOC2012/SegmentTrain"
img_path_val = "./VOCdevkit/VOC2012/SegmentVal"
label_path_train = "./VOCdevkit/VOC2012/SegmentTrainLabels"
label_path_val = "./VOCdevkit/VOC2012/SegmentValLabels"


class ImageData(Dataset):
    """Create image dataset from directory of images and  directory of labels
    """

    def __init__(self, imgdir, labeldir, transform=None):

        self.imgdir = imgdir
        self.imgfiles = os.listdir(imgdir)
        self.labdir = labeldir

        # create class bases on pixel values
        self.classes = [[0, 0, 0], [192, 224, 224], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                        [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64],
                        [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192],
                        [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0],
                        [0, 192, 128], [128, 64, 0]]
        # These pixel values are linearly dependent so we use a dot product to make them distinct
        self.indep_classes = np.dot(self.classes, [1, 10, 100])
        #And create a dictionary
        self.dictionary = dict(
            zip(self.indep_classes, [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))

    def class_enc(self, arr):
        """ Takes HxWx3 label and encodes to 1xHxW
        """
        h, w = arr.shape[0], arr.shape[1]
        arr = np.dot(arr, [1, 10, 100])
        # Create placeholder array
        res = np.zeros((h, w, 1))

        for i in range(len(arr)):
            for j in range(len(arr[i])):
                #replace pixel values with integer labels
                res[i][j] = self.dictionary[arr[i][j]]

        return res

    def random_crop(self, img, size):
        """Takes HxWxC img and returns SizexSizexC img
        """
        h, w = img.shape[0], img.shape[1]

        #pick a random valid point to be the upper left corner of the crop
        p1 = random.randint(0, h - size)
        p2 = random.randint(0, w - size)

        #return the cropped img
        return img[p1:p1 + size, p2:p2 + size, :]

    def __len__(self):
        return (len(os.listdir(self.imgdir)))

    def __getitem__(self, idx):

        #image filename
        img = self.imgfiles[idx]
        #read label and img with image filename
        lab = cv2.imread(os.path.join(self.labdir, img[:-4] + ".png"))
        img = cv2.imread(os.path.join(self.imgdir, img))

        #if the right sized crop if impossible choose a different img (this is rare)
        if img.shape[0] < 256 or img.shape[1] < 256 or lab is None:
            # if a 256x256 cant be made, select a different random image from the dataset
            rand = random.randint(0, len(os.listdir(self.imgdir)) - 1)
            return self[rand]

        # apply same random crop to label and image
        img, lab = np.array(img), np.array(lab)
        both = np.concatenate((img, lab), axis=2)
        both = self.random_crop(both, 256)
        img, lab = np.split(both, 2, axis=2)

        # make CxHxW
        img = np.swapaxes(img, 2, 1)
        img = np.swapaxes(img, 1, 0)

        # normalize
        img = img / 255
        img = tensor(img)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)

        # encode classes on label
        lab = self.class_enc(lab)

        # make CxHxW
        lab = np.swapaxes(lab, 2, 1)
        lab = np.swapaxes(lab, 1, 0)

        return img, lab

if __name__=='__main__':
    # input image index
    img = int(sys.argv[1])
    data = ImageData(img_path_train, label_path_train)
    d = data[img]

    #plot img and label side by side
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(d[0][0])
    f.add_subplot(1,2,2)
    plt.imshow(d[1][0])
    plt.show()


