import os
import shutil

#path to txt file listing filenames of segmented images
set_path = ".\VOCdevkit\VOC2012\ImageSets\Segmentation\\trainval.txt"
#path to directory of training images
src_path = ".\VOCdevkit\VOC2012\JPEGImages"
#path to destination directory
dst_path = ".\VOCdevkit\VOC2012\SegmentationImages"

def seg_data_sep():
    """Reads text file and copies segmented images to a destination directory
    """

    txtfile = open(set_path)
    seg_names = txtfile.readlines()

    for name in seg_names:
        name = name[:-1] #get rid of /n escape character
        shutil.copy(os.path.join(src_path,name+".jpg"), dst_path)

    txtfile.close()

seg_data_sep()

#number of images in train/val set
data_len = len(os.listdir(dst_path))

def train_val_sep(portion):
    """Separates Training and Validation images and labels by MOVING to new directories NOT copying

    Args:
        portion: percent to use a training. The remaining will become validation
    """
    cut = int(data_len * portion)
    test_set = os.listdir(".\VOCdevkit\VOC2012\SegmentationImages")[:cut]
    val_set = os.listdir(".\VOCdevkit\VOC2012\SegmentationImages")[cut:]

    for filename in test_set:
        shutil.move(os.path.join(".\VOCdevkit\VOC2012\SegmentationImages", filename), ".\VOCdevkit\VOC2012\SegmentTrain")
        shutil.move(os.path.join(".\VOCdevkit\VOC2012\SegmentationClass", filename[:-4]+".png"), ".\VOCdevkit\VOC2012\SegmentTrainLabels")
    for filename in val_set:
        shutil.move(os.path.join(".\VOCdevkit\VOC2012\SegmentationImages", filename), ".\VOCdevkit\VOC2012\SegmentVal")
        shutil.move(os.path.join(".\VOCdevkit\VOC2012\SegmentationClass", filename[:-4]+".png"), ".\VOCdevkit\VOC2012\SegmentValLabels")

train_val_sep(0.8)



