from SegmentNet2 import SegmentNet2
from DatasetCreate import ImageData
from TverskyLoss import tversky_loss
from LoadPretrainedVGG16 import load_pretraining
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import os
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

wandb.init(project="segmentation-net")

print("Loading Data...")
#Create Dataset Objects
trainset = ImageData("./VOCdevkit/VOC2012/SegmentTrain", "./VOCdevkit/VOC2012/SegmentTrainLabels")
valset = ImageData("./VOCdevkit/VOC2012/SegmentVal", "./VOCdevkit/VOC2012/SegmentValLabels")

BATCH_SIZE = 1

#Create DataLoader Objects
train_loader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)
print("completed.")

device = torch.device('cuda:0')

#Path to VGG16 state dictionary
PATH = "./"
VGG_PATH = os.path.join(PATH,"vgg16-397923af.pth")

print("Initializing Network...")
def net_init(model):
    """Initialize model weights to Gaussian Distribution
    """
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        model.weight.data.uniform_(0.0, 1.0)
        if model.bias is not None:
            model.bias.data.fill_(0.0)

def param_freeze(model, param_num):
    """ Freeze weigths and bias within the model
        param_num: number of parameters at beginning of network to freeze
    """
    count = 0
    for p in model.parameters():
      count +=1
      if count <= param_num:
        p.requires_grad_(False)
    return model

#Initialize Model
net = SegmentNet2().type(torch.cuda.FloatTensor)
#Apply Initialization to weights
net.apply(net_init)
#load pretrained weights
net_dict = net.state_dict()
pretrained_dict = torch.load(VGG_PATH)
transfer_dict = load_pretraining(net_dict, pretrained_dict)
net.load_state_dict(transfer_dict)
#freeze VGG16 part of the net
net = param_freeze(net, 26)

wandb.watch(net)
print("completed")

EPOCHS = 25

def train(net):

    print("Training Beginning")

    optimizer = optim.SGD(net.parameters(), lr=1e-2,  weight_decay=5e-4, momentum=0.9)
    weights = torch.tensor([1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=1, reduction='mean').cuda()
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

    for epoch in range(EPOCHS):

        print("Training...")
        net.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(tqdm(train_loader)):

            X, label = x.type(torch.cuda.FloatTensor), y.type(torch.cuda.FloatTensor)

            optimizer.zero_grad()

            output_label = net(X)

            label = label.squeeze(1)
            #loss1 = criterion(output_label.type(torch.cuda.FloatTensor), label.type(torch.cuda.LongTensor))
            loss = tversky_loss(true=label.type(torch.cuda.LongTensor), logits=output_label.type(torch.cuda.FloatTensor), alpha=0.5, beta=0.5)



            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:
                print("Epoch: %d    Training Loss: %.5f\n" % (epoch+1, running_loss))
                wandb.log({"Training Loss": running_loss})
                running_loss = 0.0
                torch.cuda.empty_cache()

        print("Running Validation...")
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_acc = 0.0
            full_val_loss = 0.0
            for i, (x, y) in enumerate(tqdm(val_loader)):

                X, label = x.type(torch.cuda.FloatTensor), y.type(torch.cuda.FloatTensor)
                output_label = net(X)

                label = label.squeeze(1)
                #val_loss = criterion(output_label.type(torch.cuda.FloatTensor), val_lab.type(torch.cuda.LongTensor))
                val_loss = tversky_loss(true=label.type(torch.cuda.LongTensor), logits=output_label.type(torch.cuda.FloatTensor), alpha=0.5, beta=0.5)


                acc = (tversky_loss(label.type(torch.cuda.LongTensor), output_label.type(torch.cuda.FloatTensor), eps=1e-4, alpha=0.5, beta=0.5)-1)*-1

                running_val_loss += val_loss.item()
                running_val_acc += acc
                full_val_loss += val_loss.item()
                if i % 5 == 4:
                    wandb.log({"Val Loss": running_val_loss, "Val Acc": running_val_acc})
                    print("  Validation Loss: %.5f     Validation Acc: %.5f\n" % (running_val_loss, running_val_acc))
                    running_val_loss = 0.0
                    running_val_acc = 0.0
                    torch.cuda.empty_cache()
        #scheduler.step(full_val_loss)

if __name__=='__main__':
    train(net)
    torch.save(net.state_dict(), "./TrainedNet1_2")
