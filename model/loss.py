import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def CrossEntropy(output, target):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    return loss
# Custom focal loss implementation


def FocalLoss(output, target):
    criterion = FocalLoss()
    loss = criterion(output, target)
    return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss()(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss
