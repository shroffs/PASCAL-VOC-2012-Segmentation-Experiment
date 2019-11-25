import os
import sys
import numpy as np
import cv2
from torch.utils.data import Dataset
np.set_printoptions(threshold=sys.maxsize)

img_path_train = "./VOCdevkit/VOC2012/SegmentTrain"
img_path_val = "./VOCdevkit/VOC2012/SegmentVal"
label_path_train = "./VOCdevkit/VOC2012/SegmentTrainLabels"
label_path_val = "./VOCdevkit/VOC2012/SegmentValLabels"

class ImageData(Dataset):
    """Create image dataset from folder

    """
    def __init__(self, imgdir, transform=None):
        """Args:
                imgdir: Directory of images
        """
        self.imgdir = imgdir
        self.files = os.listdir(imgdir)


    def __len__(self):
        return (len(os.listdir(self.imgdir)))

    def __getitem__(self, idx):
        img = self.files[idx]
        #read jpg
        img = cv2.imread(os.path.join(self.imgdir, img))
        # swap axes for CxHxW array
        img = np.swapaxes(img, 2, 0)
        return img


class LabelData(Dataset):
    """Create label dataset from folder. Create Label Map:  3xHxW to 1xHxW

    """
    def __init__(self, imgdir, transform=None):

        self.imgdir = imgdir
        self.files = os.listdir(imgdir)

        #create class bases on pixel values
        self.classes = [[192, 224, 224],[0, 0, 0], [0, 0, 128],
                   [0, 128, 0], [0, 128, 128], [128, 0, 0],
                   [128, 0, 128], [128, 128, 0], [128, 128, 128],
                   [128, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                   [128, 0, 64], [128, 0, 192], [128, 128, 192],
                   [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128],
                   [128, 64, 0]]
        #These pixel values are linearly dependent so we use a dot product to make them distinct
        self.indep_classes = np.dot(self.classes, [1, 10, 100])
        self.dictionary = dict(zip(self.indep_classes, range(21)))

    def class_enc(self, arr):
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
        return res

    def __len__(self):
            return (len(os.listdir(self.imgdir)))

    def __getitem__(self, idx):
        img = self.files[idx]
        # read image
        img = cv2.imread(os.path.join(self.imgdir, img))
        # convert to numpy array
        img = np.array(img, dtype=int)
        # encode
        img = self.class_enc(img)
        # swap axis for CxHxW array
        img = np.swapaxes(img, 2, 0)
        return img