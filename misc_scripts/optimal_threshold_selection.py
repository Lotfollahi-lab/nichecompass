import numpy as np
import copy
import torch
from scipy.special import expit
from sklearn import metrics
from sklearn.metrics  import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from deeplinc.train.metrics import get_optimal_cls_threshold_and_accuracy

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

thresh, acc = get_optimal_cls_threshold_and_accuracy(Z, edges_test_pos, edges_test_neg)
print(thresh)
print(acc)

A_rec_probs = torch.sigmoid(torch.mm(Z, Z.T))

print("\n", f"A_rec_probs: {A_rec_probs}", "\n")

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
all_labels = torch.hstack([torch.ones(len(pred_probs_pos_labels)),
                           torch.zeros(len(pred_probs_neg_labels))])

print(f"All labels: {all_labels}", "\n")

# Calculate accuracies for all thresholds and store best accuracy and threshold
all_acc_score = {}
max_acc_score = 0
optimal_threshold = 0
for threshold in np.arange(0.01, 1, 0.005):
    pred_probs_all_labels = np.hstack([pred_probs_pos_labels,
                                       pred_probs_neg_labels])
    preds_all_labels = (pred_probs_all_labels>threshold).astype("int")
    acc_score = accuracy_score(all_labels, preds_all_labels)
    all_acc_score[threshold] = acc_score
    if acc_score > max_acc_score:
        max_acc_score = acc_score
        optimal_threshold = threshold

print(f"Max accuracy: {max_acc_score}")
print(f"Optimal threshold: {optimal_threshold}")
print("")

for i in range(0, A_rec_probs.shape[0]):
    A_rec_probs[i,i] = 0 # Set diagonal elements to 0

A_rec = copy.deepcopy(A_rec_probs)
A_rec = (A_rec>optimal_threshold).int()

print(f"A_rec: {A_rec}", "\n")

def reduce_edges_per_node(A_rec_probs, A_rec, max_edge_target, type):
    A_rec_tmp = copy.deepcopy(A_rec_probs)
    for node in range(0, A_rec_tmp.shape[0]):
        tmp = A_rec_tmp[node,:]
        A_rec_tmp[node,:] = (A_rec_tmp[node,:] >= np.sort(tmp)[-max_edge_target]).int()
    print(f"A_rec tmp: {A_rec_tmp}", "\n")
    A_rec_new = A_rec + A_rec_tmp
    print(f"A_rec new: {A_rec_new}", "\n")
    # Mark edges that have been recreated and are among the top 2 node edges for
    # one of the nodes
    A_rec_new = (A_rec_new == 2).int()
    print(f"A_rec new: {A_rec_new}", "\n")
    # Make adjacency matrix symmetric
    A_rec_new = A_rec_new + A_rec_new.T 
    print(f"A_rec new: {A_rec_new}", "\n")
    # keep edges that are among the top 2 node edges for one of the nodes
    if type == "soft": # union
        A_rec_new = (A_rec_new != 0).int()
    # keep edges that are among the top 2 node edges for both of the nodes
    elif type == "hard": # intersection
        A_rec_new = (A_rec_new == 2).int()
    return A_rec_new

A_rec_soft = reduce_edges_per_node(A_rec_probs, A_rec, 2, "soft")
A_rec_hard = reduce_edges_per_node(A_rec_probs, A_rec, 2, "hard")

print(f"A_rec soft: {A_rec_soft}", "\n")
print(f"A_rec hard: {A_rec_hard}", "\n")

assert (A_rec==A_rec.T).all()
assert (A_rec_soft==A_rec_soft.T).all()
assert (A_rec_hard==A_rec_hard.T).all()