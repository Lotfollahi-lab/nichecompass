import numpy as np
import copy
import torch
import sys
from scipy.special import expit
from sklearn import metrics
from sklearn.metrics  import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from deeplinc.train.metrics import get_eval_metrics

# 5 nodes, 2 dimensions latent space simulation
torch.manual_seed(0)
Z = torch.randn(5, 2)

# Single direction of edges labels (without diagonal edges)
"""
edges_test_pos = np.array([[0, 3],
                           [1, 2],
                           [1, 4],
                           [2, 4]])
edges_test_neg = np.array([[0, 1],
                           [0, 2],
                           [0, 4],
                           [2, 3]])  
"""      

edges_test_pos = np.array([[0, 3],
                           [1, 2],
                           [1, 4],
                           [2, 3]])
edges_test_neg = np.array([[0, 1],
                           [0, 2],
                           [0, 4],
                           [2, 4]])

A_rec_probs = torch.sigmoid(torch.mm(Z, Z.T))

roc, ap, acc = get_eval_metrics(Z, edges_test_pos, edges_test_neg)
print(roc)
print(ap)
print(acc)
sys.exit(1)

# Collect predictions for each label (positive vs negative edge) separately
pred_probs_pos_labels = []
for edge in edges_test_pos:
    pred_probs_pos_labels.append(A_rec_probs[edge[0], edge[1]])
pred_probs_neg_labels = []
for edge in edges_test_neg:
    pred_probs_neg_labels.append(A_rec_probs[edge[0], edge[1]])

print(f"Predicted probabilities for positive labels: {pred_probs_pos_labels}")   
print(f"Predicted probabilities for negative labels: {pred_probs_neg_labels}")   
print("")

# Create tensor of ground truth labels
pred_probs_all_labels = torch.hstack([torch.tensor(pred_probs_pos_labels),
                                      torch.tensor(pred_probs_neg_labels)])
preds_all_labels = (pred_probs_all_labels>0.5).int()
all_labels = torch.hstack([torch.ones(len(pred_probs_pos_labels)),
                           torch.zeros(len(pred_probs_neg_labels))])

roc_score = roc_auc_score(all_labels, pred_probs_all_labels)
ap_score = average_precision_score(all_labels, pred_probs_all_labels)
acc_score_thresh_50percent = accuracy_score(all_labels, preds_all_labels)

print(roc_score)
print(ap_score)
print(acc_score_thresh_50percent)