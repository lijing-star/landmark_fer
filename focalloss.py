
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
    """
    def __init__(self, loss_type, class_num, smooth_factor=30):
        super(FocalLoss, self).__init__()

        self.smooth_factor = smooth_factor
        self.class_num = class_num

        self.loss_type = loss_type
        if self.loss_type == 'KL':
            self.loss_func = torch.nn.KLDivLoss(reduction='none').to(device)
        elif self.loss_type == 'CE':
            self.loss_func = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        else:
            raise ValueError('no such loss function ...')

    def forward(self, logits, labels, targets):
        if self.loss_type == 'KL':
            if targets == None:
                log_probs = F.log_softmax(logits, dim=0)
                kl_probs = F.softmax(labels * self.smooth_factor, dim=0)
                loss = self.loss_func(log_probs, kl_probs)
            else:
                log_probs = F.log_softmax(logits, dim=1)
                kl_probs = F.softmax(labels * self.smooth_factor, dim=1)
                loss = self.loss_func(log_probs, kl_probs)
        elif self.loss_type == 'CE':
            loss = self.loss_func(logits, targets.long())
        else:
            raise ValueError('no such loss function ...')

        return loss.mean()


if __name__ == '__main__':
    input = torch.rand(2, 8)
    label = torch.Tensor([1, 5]).long()
    weight = torch.Tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1])

    criterion = FocalLoss('CE', 8, weight)
    loss = criterion(input, label)

    criterion = FocalLoss('KL', 8, weight)
    loss = criterion(input, label)

    print(loss.item())
