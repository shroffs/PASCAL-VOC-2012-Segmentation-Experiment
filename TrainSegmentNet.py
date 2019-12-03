from SegmentNet import SegmentNet
from DatasetCreate import ImageData
from TverskyLoss import tversky_loss
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import os
from tqdm import tqdm
import wandb

wandb.init(project="segmentation-net")

print("Loading Data...")
trainset = ImageData("./VOCdevkit/VOC2012/SegmentTrain", "./VOCdevkit/VOC2012/SegmentTrainLabels")
valset = ImageData("./VOCdevkit/VOC2012/SegmentVal", "./VOCdevkit/VOC2012/SegmentValLabels")

BATCH_SIZE = 5

train_loader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)
print("completed.")

device = torch.device('cuda:0')

PATH = "./"
VGG_PATH = os.path.join(PATH,"vgg16_bn-6c64b313.pth")

print("Initializing Network...")
def net_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        model.weight.data.uniform_(0.0, 0.1)
        if model.bias is not None:
            model.bias.data.fill_(0.0)

net = SegmentNet().type(torch.cuda.FloatTensor)
net.apply(net_init)
net.load_state_dict(torch.load(VGG_PATH), strict=False)
wandb.watch(net)
print("completed")

EPOCHS = 1

def train(net):
    net.train()
    print("Training Beginning")
    optimizer = optim.SGD(net.parameters(), lr=5e-5, momentum=0.99, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().cuda()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (x, y) in enumerate(tqdm(train_loader)):

            X, label = x.type(torch.cuda.FloatTensor), y.type(torch.cuda.FloatTensor)
            
            optimizer.zero_grad()

            output_label = net(X)

            label = label.squeeze(1)
            t_loss = tversky_loss(label.type(torch.cuda.LongTensor), output_label.type(torch.cuda.FloatTensor), eps=1e-4, alpha=0.05, beta=2)
            ce_loss = 0.1*criterion(output_label.type(torch.cuda.FloatTensor), label.type(torch.cuda.LongTensor))
            loss = ce_loss.add(t_loss)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:
                print("Epoch: %d    Training Loss: %.5f\n" % (epoch+1, running_loss/5))
                wandb.log({"Training Loss": running_loss/5})
                running_loss = 0.0
                torch.cuda.empty_cache()

        print("Running Validation...")
        running_val_loss = 0.0
        running_val_acc = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_loader)):

                val_img, val_lab = x.type(torch.cuda.FloatTensor), y.type(torch.cuda.FloatTensor)
                output_label = net(val_img)
                val_lab = val_lab.squeeze(1)

                t_loss = tversky_loss(val_lab.type(torch.cuda.LongTensor), output_label.type(torch.cuda.FloatTensor), eps=1e-4, alpha=0.05, beta=2)
                ce_loss = 0.1*criterion(output_label.type(torch.cuda.FloatTensor), val_lab.type(torch.cuda.LongTensor))
                val_loss = ce_loss.add(t_loss)

                acc = (tversky_loss(val_lab.type(torch.cuda.LongTensor), output_label.type(torch.cuda.FloatTensor), eps=1e-4, alpha=0.5, beta=0.5)-1)*-1

                running_val_acc += val_loss.item()
                running_val_acc += acc
                if i % 5 == 4:
                    wandb.log({"Val Loss": running_val_loss/5, "Val Acc": running_val_acc/5})
                    print("  Validation Loss: %.5f     Validation Acc: %.5f\n" % (running_val_loss, running_val_acc))
                    running_val_loss = 0.0
                    running_val_acc = 0.0
                    torch.cuda.empty_cache()


train(net)
#16:14 min/epoch

torch.save(net.state_dict(), "./TrainedNet1")