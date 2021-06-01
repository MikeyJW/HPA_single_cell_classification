'''
A set of custom loss functions for network training.
Loss functions borrowed from this excellent kaggle post...
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
'''

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


# Focal Loss (to cope with highly imbalanced dataset)

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # First compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE   
        return focal_loss

    
# Tversky loss (allows weighting of penalisation for false positive 
# or false negatives through alpha and beta)
    
#ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        return 1 - Tversky
    