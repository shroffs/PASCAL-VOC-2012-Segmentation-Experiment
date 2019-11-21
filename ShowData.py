import os
import sys
import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np

test_path = '.\VOCdevkit\VOC2012\SegmentTrain'
label_path = '.\VOCdevkit\VOC2012\SegmentTrainLabels'

def show_data(n):
    """Shows image and label and prints the shapes of image and labels

    Args:
        n: index of image in directory
    """
    img_filename= os.listdir(test_path)[n]
    img = cv2.imread(os.path.join(test_path,img_filename))

    label_filename= os.listdir(label_path)[n]
    label = cv2.imread(os.path.join(label_path,label_filename))

    cv2.imshow('label',label)
    cv2.imshow('img',img)
    print(img.shape, label.shape)
    cv2.waitKey(0)


show_data(6)

def show_masks(n):
    """Show label mask apples to image

    Args:
        n: index of image in directory

    """

    label_filename = os.listdir(label_path)[n]
    label = imageio.imread(os.path.join(label_path, label_filename))

    img_filename= os.listdir(test_path)[n]
    img = cv2.imread(os.path.join(test_path,img_filename))

    #for each mask
    for i in range(label.shape[2]):
        mask = label[:,:,i]
        plt.imshow(mask)
        plt.show()

show_data(5)
show_masks(5)
