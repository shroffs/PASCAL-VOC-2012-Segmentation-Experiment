from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from SegmentNet import SegmentNet
from BabySegmentNet import BabySegmentNet
from tqdm import tqdm
import sys
import torch.nn as nn
from torch import tensor

print("Loading Data...")
X_train = np.load("./VOCdevkit/VOC2012/Training_Enc.npy", allow_pickle=True)
# X_val = np.load("./VOCdevkit/VOC2012/Val_Enc.npy")

Y_train = np.load("./VOCdevkit/VOC2012/Training_Labels_Enc.npy")
# Y_val = np.load("./VOCdevkit/VOC2012/Val_Labels_Enc.npy")

# 1min:39sec:23ms to load all data

trainset = list(zip(X_train, Y_train))

X_train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

device = torch.device(0)


def net_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        model.weight.data.uniform_(0.0, 1.0)
        if model.bias is not None:
            model.bias.data.fill_(0.0)


print("Initializing Network")
net = BabySegmentNet().type(torch.FloatTensor)
net.apply(net_init)

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
    optimizer = optim.RMSprop(net.parameters(), lr=0.01, momentum=0.9, eps=1e-4)
    # optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=20)
    # criterion = nn.CrossEntropyLoss(weight=tensor([1.0,1.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0])).cuda()
    for epoch in range(EPOCHS):
        loss_track = 0.0
        for data in enumerate(tqdm(X_train_loader)):
            X, label = data[1][0].type(torch.FloatTensor), data[1][1].type(torch.FloatTensor)

            optimizer.zero_grad()

            output_label = net(X)

            label = label.squeeze(1)
            # loss = criterion(output_label.type(torch.cuda.FloatTensor), label.type(torch.cuda.LongTensor))
            loss = jaccard_loss(label.type(torch.LongTensor), output_label.type(torch.FloatTensor), eps=1e-4)
            loss.backward()

            # clip = 1
            # torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            loss_track += loss.item()
            if data[0] % 20 == 1:
                print("Epoch: %d    Loss: %.5f\n" % (epoch + 1, loss_track))
                loss_track = 0.0


train(net)
# 16:14 min/epoch

torch.save(net.state_dict(), "./TrainedNet3-1EPOCH")


