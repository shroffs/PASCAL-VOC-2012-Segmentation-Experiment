# PASCAL-VOC-2012-Segmentation-Experiement
Using a FCN/Unet style network to segment images.  [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) segementation dataset used as training data.
____
### The Purpose: Larger Networks and Cloud GPUs
The initial purpose of this was to exercise making larger more complex networks than plain old CNNs I have made in the past. As a bonus it turns out my GPU did not have the RAM to hanlde such a network and the training that comes with it... meaning I had to step into the world of cloud GPUs.
___
### The Scripts
The code was initially intended to run locally, meaning i/o was sloppy and runtime was a matter of patients. When I realized I would have to use a cloud GPU, I made the code more modular in a way that made more sense and saved more time. The order this were intended to run looks like this.

1. Download the dataset to working directory
 ```
 wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
 ```
 2. Extract the dataset
 ```
 python3 DataExtract.py
 ````
 3. Separate Data that is labeled from segmentation and separate training and validation data
 ```
 python3 DataSeparate.py
 ```
 4. DatasetCreate.py and SegmentNet.py are just classes that are called in actuall training script
 5. Train the model
 ```
 python3 TrainSegmentNet.py
 ```
 ___
 ### Network Architecture
 The architecture I chose for this task is a Fully-Convolutional Network with UNet skip connections and residual layers. This architecture was mainly choosen to be adventurous and explore the pytorch library but if it performs have decently that would be great as well. In the spirit of being adventurous, we also use [Jaccard Loss](https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py).
 
 ___
 ### Cloud GPUs
 Linode will be used. From my understanding the process goes something like this.
 1. Create a CPU linode (Ubuntu 18.04). We'll call this linode-A
 2. Create a Volume that can fit you dataset and mount it to linode-A
 3. WIP :)
