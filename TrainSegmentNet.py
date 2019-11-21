from torch.utils.data import DataLoader
from DataFormat import ImageData
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.tensor as tensor
import torch
from SegmentNet import SegmentNet
from tqdm import tqdm

X_Dataset = ImageData("./VOCdevkit/VOC2012/SegmentTrain")
X_train = X_Dataset.dataset
print(X_train.shape)
#X_val = ImageData("./VOCdevkit/VOC2012/SegmentVal")

"""
Y_train = np.load("./VOCdevkit/VOC2012/Training_Labels_Enc.npy")
#Y_val = np.load("./VOCdevkit/VOC2012/Val_Labels_Enc.npy")

#1min:39sec:23ms to load all data

trainset = list(zip(X_train,Y_train))

X_train_loader = DataLoader(trainset, batch_size=1)

net = SegmentNet()

optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(weight=tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,]))
EPOCHS = 1

for epoch in tqdm(range(EPOCHS)):
    for data in X_train_loader:
        X, label = data[0].type(torch.FloatTensor), data[1].type(torch.FloatTensor)

        optimizer.zero_grad()

        output_label = net(X)
        loss = criterion(output_label, label)

        loss.backward()
        optimizer.step()
        break
    print(loss)

"""


