import os
import numpy as np
import cv2
from torch.utils.data import Dataset

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
        self.dataset = []
        for img in os.listdir(self.imgdir):
            img = cv2.imread(os.path.join(self.imgdir,img))
            #make images 512x512
            img = cv2.resize(img,(512,512))
            #swap axes for CxHxW array
            img = np.swapaxes(img,2,0)
            self.dataset.append(img)
        self.dataset = np.array(self.dataset)

    def __len__(self):
        return (len(os.listdir(self.imgdir)), 512, 512, 3)

    def __getitem__(self, idx):
        return np.array(self.dataset[idx])


class LabelData(Dataset):
    """Create label dataset from folder. Hot-Encodes 500x500x3 to 500x500x21

    """
    def __init__(self, labeldir, transform=None):
        self.labeldir = labeldir
        self.dataset = []

        #create class bases on pixel values
        classes = [[192, 224, 224],[0, 0, 0], [0, 0, 128],
                   [0, 128, 0], [0, 128, 128], [128, 0, 0],
                   [128, 0, 128], [128, 128, 0], [128, 128, 128],
                   [128, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                   [128, 0, 64], [128, 0, 192], [128, 128, 192],
                   [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128],
                   [128, 64, 0]]
        #These pixel values are linearly dependent so we use a dot product to make them distinct
        indep_classes = np.dot(classes, [1, 10, 100])
        dictionary = dict(zip(indep_classes, range(21)))

        def class_enc(arr):

            arr = np.dot(arr, [1, 10, 100])

            # Create higher dimension placeholder array
            res = np.zeros((512, 512, 1))

            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    try:
                        res[i][j] = dictionary[arr[i][j]]
                    except KeyError:
                        # The the pixel value does not belong to a class make it border class
                        res[i][j] = 0
            return res

        """
        #Create one hot vectors
        hot_vectors = np.eye(len(indep_classes), dtype='b')
        #Create diction to take modified class as key and one hot vector as a value
        dictionary = dict(zip(indep_classes, hot_vectors))
        
        def one_hot(arr):
            convert segmented RGB image to 500x500x21 one hot encoded array.
            This function will only work if pixel values in image are segmented and only
            have RGB values of the classes above

            Args:
                arr: image being input
            
            arr = np.dot(arr, [1, 10, 100])

            #Create higher dimension placeholder array
            res = np.zeros((512, 512, 21), dtype='b')

            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    try:
                        res[i][j] = dictionary[arr[i][j]]
                    except KeyError:
                        #The the pixel value does not belong to a class make it the 0 vector
                        res[i][j] = np.zeros(21, dtype='b')
            return res
        """

        for img in os.listdir(self.labeldir):
            #read image
            img = cv2.imread(os.path.join(self.labeldir, img))
            #resize to 512x512x3
            img = cv2.resize(img, (512, 512))
            #convert to numpy array
            img = np.array(img, dtype=int)
            #encode
            img = class_enc(img)
            #swap axis for CxHxW array
            img = np.swapaxes(img, 2, 0)
            #add to dataset
            self.dataset.append(img)
        self.dataset = np.array(self.dataset)

        def __len__(self):
            return (len(os.listdir(self.imgdir)), 512, 512, 21)

        def __getitem__(self, idx):
            return np.array(self.dataset[idx])

#trainingset = ImageData(img_path_train)
#np.save('./VOCdevkit/VOC2012/Training_Enc',trainingset.dataset, allow_pickle=True)
#print(trainingset.dataset.shape)

#valset = ImageData(img_path_val)
#np.save('./VOCdevkit/VOC2012/Val_Enc',valset.dataset, allow_pickle=True)
#print(valset.dataset.shape)

#labels = LabelData(label_path_train)
#np.save('./VOCdevkit/VOC2012/Training_Labels_Enc',labels.dataset, allow_pickle=True)
#print(labels.dataset.shape)

#val_labels = LabelData(label_path_val)
#np.save('./VOCdevkit/VOC2012/Val_Labels_Enc', val_labels.dataset, allow_pickle=True)
#print(val_labels.dataset.shape)

