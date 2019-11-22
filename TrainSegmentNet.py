from torch.utils.data import DataLoader
from DataFormat import ImageData
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.tensor as tensor
import torch
from SegmentNet import SegmentNet
from tqdm import tqdm
import sys

X_train = np.load("./VOCdevkit/VOC2012/Training_Enc.npy", allow_pickle=True)
#X_val = np.load("./VOCdevkit/VOC2012/Val_Enc.npy")

Y_train = np.load("./VOCdevkit/VOC2012/Training_Labels_Enc.npy")
#Y_val = np.load("./VOCdevkit/VOC2012/Val_Labels_Enc.npy")

#1min:39sec:23ms to load all data

trainset = list(zip(X_train,Y_train))

X_train_loader = DataLoader(trainset, batch_size=1)

device = torch.device(0)

net = SegmentNet().type(torch.cuda.HalfTensor).cuda()

EPOCHS = 1

def train(net):
    print("Training Beginning")
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(weight=tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,])).cuda()
    for epoch in tqdm(range(EPOCHS)):
        for data in X_train_loader:
            X, label = data[0].type(torch.cuda.HalfTensor), data[1].type(torch.cuda.HalfTensor)
            X, label = X.to(device), label.to(device)
            
            optimizer.zero_grad()

            #small GPU: only 3GiB :(
            try:
                output_label = net(X)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("No more memory: retrying", sys.stdout)
                    sys.stdout.flush()
                    for p in net.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    output_label = net(X)
                else:
                    raise e

            label = label.squeeze(1)
            loss = criterion(output_label.type(torch.cuda.FloatTensor), label.type(torch.cuda.LongTensor))
    
            loss.backward()
            optimizer.step()
        print("Epoch: %d    Loss: %f" % (epoch, loss))
        
train(net)


