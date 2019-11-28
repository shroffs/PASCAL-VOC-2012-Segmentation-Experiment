import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
from torch import tensor

img_path_train = "./VOCdevkit/VOC2012/SegmentTrain"
img_path_val = "./VOCdevkit/VOC2012/SegmentVal"
label_path_train = "./VOCdevkit/VOC2012/SegmentTrainLabels"
label_path_val = "./VOCdevkit/VOC2012/SegmentValLabels"

class ImageData(Dataset):
    """Create image dataset from folder

    """
    def __init__(self, imgdir, labeldir, transform=None):

        self.imgdir = imgdir
        self.imgfiles = os.listdir(imgdir)
        self.labdir = labeldir
        self.labfiles = os.listdir(labeldir)

        # create class bases on pixel values
        self.classes = [[0, 0, 0],[192, 224, 224],[0, 0, 128],[0, 128, 0],[0, 128, 128],
                        [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64],
                        [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192],
                        [128, 128, 64],  [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0] ,
                        [0, 192, 128], [128, 64, 0]]
        # These pixel values are linearly dependent so we use a dot product to make them distinct
        self.indep_classes = np.dot(self.classes, [1, 10, 100])
        self.dictionary = dict(zip(self.indep_classes, [0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]))

    def class_enc(self, arr):
        """ Takes HxWx3 label and encodes to 1xHxW
        """
        h, w = arr.shape[0], arr.shape[1]
        arr = np.dot(arr, [1, 10, 100])
        # Create higher dimension placeholder array
        res = np.zeros((h, w, 1))

        for i in range(len(arr)):
            for j in range(len(arr[i])):
                try:
                    res[i][j] = self.dictionary[arr[i][j]]
                except KeyError:
                    # The the pixel value does not belong to a class make it border class
                    res[i][j] = 0
                    print(i, j, arr[i][j])
        return res


    def __len__(self):
        return (len(os.listdir(self.imgdir)))

    def __getitem__(self, idx):

        img = self.imgfiles[idx]
        #read jpg
        img = cv2.imread(os.path.join(self.imgdir, img))
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_NEAREST)
        # swap axes for CxHxW array
        img = np.array(img)
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)



        lab = self.labfiles[idx]
        # read image
        lab = cv2.imread(os.path.join(self.labdir, lab))
        lab = cv2.resize(lab, (512, 512), interpolation=cv2.INTER_NEAREST)
        # convert to numpy array
        lab = np.array(lab)
        # encode
        lab = self.class_enc(lab)
        # swap axis for CxHxW array
        lab = np.swapaxes(lab, 1, 2)
        lab = np.swapaxes(lab, 0, 1)

        return img, lab