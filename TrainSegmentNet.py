from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from SegmentNet import SegmentNet
from tqdm import tqdm
import sys
from torchsummary import summary

print("Loading Data...")
X_train = np.load("./VOCdevkit/VOC2012/Training_Enc.npy", allow_pickle=True)
#X_val = np.load("./VOCdevkit/VOC2012/Val_Enc.npy")

Y_train = np.load("./VOCdevkit/VOC2012/Training_Labels_Enc.npy")
#Y_val = np.load("./VOCdevkit/VOC2012/Val_Labels_Enc.npy")

#1min:39sec:23ms to load all data

trainset = list(zip(X_train,Y_train))

X_train_loader = torch.utils.data.DataLoader(trainset, batch_size=1)

device = torch.device(0)

def net_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        model.weight.data.uniform_(0.0, 1.0)
        if model.bias is not None:
            model.bias.data.fill_(0.0)

print("Initializing Network")
net = SegmentNet().type(torch.cuda.HalfTensor).cuda()
net.apply(net_init)

EPOCHS = 1

def dice_loss(true, logits, eps=1e-7): #https://github.com/kevinzakka/pytorch-goodies.git
    """"Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
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
        probas = torch.nn.Softmax(dim=1)(logits)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def train(net):
    print("Training Beginning")
    optimizer = optim.Adam(net.parameters(), lr=0.005, eps=1e-4)
    #criterion = nn.CrossEntropyLoss(weight=tensor([3.0,1.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0])).cuda()
    for epoch in range(EPOCHS):
        loss_track = 0.0
        for data in enumerate(tqdm(X_train_loader)):
            X, label = data[1][0].type(torch.cuda.HalfTensor), data[1][1].type(torch.cuda.HalfTensor)

            
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
            #loss = criterion(output_label.type(torch.cuda.FloatTensor), label.type(torch.cuda.LongTensor))
            loss = dice_loss(label.type(torch.cuda.LongTensor), output_label.type(torch.cuda.FloatTensor), eps=1e-3)
            loss.backward()


            clip = 0.5
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()



            loss_track += loss.item()
            if data[0] % 20 == 1:
                print("Epoch: %d    Loss: %.5f\n" % (epoch+1, loss_track))
                loss_track = 0.0
        
train(net)
#16:14 min/epoch

torch.save(net.state_dict(), "./TrainedNet2-1EPOCH")


