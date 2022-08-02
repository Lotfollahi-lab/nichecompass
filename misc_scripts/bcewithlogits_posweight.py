from turtle import pos
import torch
import torch.nn as nn

bce = torch.nn.functional.binary_cross_entropy_with_logits

pos_weight = torch.FloatTensor([5])
preds =  torch.FloatTensor([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
labels = torch.FloatTensor([0, 0, 0, 1, 1, 1])
loss = bce(preds, labels, reduction="none", pos_weight=pos_weight)
print(loss)
