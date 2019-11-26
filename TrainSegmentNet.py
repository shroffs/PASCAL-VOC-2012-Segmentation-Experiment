from torch.utils.data import DataLoader
from SegmentNet import SegmentNet
from DatasetCreate import ImageData
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from torch import tensor
import numpy as np
import sys
from tqdm import tqdm

import wandb
wandb.init(project="segmentation-net")

print("Loading Data...")
trainset = ImageData("./VOCdevkit/VOC2012/SegmentTrain", "./VOCdevkit/VOC2012/SegmentTrainLabels")
valset = ImageData("./VOCdevkit/VOC2012/SegmentVal", "./VOCdevkit/VOC2012/SegmentValLabels")

train_loader = data.DataLoader(trainset, batch_size=1, shuffle=True)
val_loader = data.DataLoader(valset, batch_size=1, shuffle=True)
print("completed.")

device = torch.device(0)


def net_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        model.weight.data.uniform_(0.0, 0.05)
        if model.bias is not None:
            model.bias.data.fill_(0.0)

print("Initializing Network...")
net = SegmentNet().type(torch.cuda.FloatTensor).cuda()
net.apply(net_init)
wandb.watch(net)
print("completed")

EPOCHS = 1

def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = nn.Softmax(dim=1)(logits)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def train(net):
    print("Training Beginning")
    optimizer = optim.Adam(net.parameters(), lr=0.1, eps=1e-4)
    #optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=20)
    #criterion = nn.CrossEntropyLoss(weight=tensor([1.0,1.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0])).cuda()
    for epoch in range(EPOCHS):
        loss_track = 0.0
        for data in enumerate(tqdm(train_loader)):
            X, label = data[1][0].type(torch.cuda.FloatTensor), data[1][1].type(torch.cuda.FloatTensor)
            
            optimizer.zero_grad()

            #small GPU: only 3GiB :( Cloud GPU :)
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
            #loss = criterion(output_label.type(torch.cuda.FloatTensor), label.type(torch.cuda.LongTensor))
            loss = jaccard_loss(label.type(torch.cuda.FloatTensor), output_label.type(torch.cuda.FloatTensor), eps=1e-4)
            loss.backward()


            #clip = 1
            #torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()



            loss_track += loss.item()
            if data[0] % 20 == 1:
                print("Epoch: %d    Training Loss: %.5f\n" % (epoch+1, loss_track))
                wandb.log({"Training Loss": loss_track})
                loss_track = 0.0

        print("Running Validation...")
        track_val_loss = 0.0
        with torch.no_grad():
            for val in enumerate(tqdm(val_loader)):
                val_img, val_lab = val[1][0].type(torch.cuda.FloatTensor), val[1][1].type(torch.cuda.FloatTensor)
                output_label = net(val_img)
                val_lab = val_lab.squeeze(1)
                val_loss = jaccard_loss(val_lab.type(torch.cuda.LongTensor), output_label.type(torch.cuda.FloatTensor), eps=1e-4)
                track_val_loss += val_loss
        track_val_loss = track_val_loss/26.4
        print("Validation Loss: %.5f\n" % track_val_loss)
        wandb.log({"Validation Loss": track_val_loss})


train(net)
#16:14 min/epoch

torch.save(net.state_dict(), "./TrainedNet1")
