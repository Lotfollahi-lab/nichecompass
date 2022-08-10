import copy

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import torch
from matplotlib.ticker import MaxNLocator


def get_eval_metrics(
        adj_rec_probs: torch.Tensor,
        pos_edge_label_index: torch.Tensor,
        neg_edge_label_index: torch.Tensor,
        debug: bool = False):
    """
    Get the evaluation metrics for a (balanced) sample of positive and negative edges.

    Parameters
    ----------
    adj_rec_probs:
        Tensor containing reconstructed adjacency matrix with edge probabilities.
    pos_edge_label_index:
        Tensor containing node indices of positive edges.
    neg_edge_label_index:
        Tensor containing node indices of negative edges.
    Returns
    ----------
    auroc_score:
        Area under the receiver operating characteristic curve.
    auprc_score:
        Area under the precision-recall curve.
    best_acc_score:
        Accuracy under optimal classification threshold.
    best_f1_score:
        F1 score under optimal classification threshold.
    """
    # Collect predictions for each label (positive vs negative edge) separately
    pred_probs_pos_edge_label = []
    for edge in zip(pos_edge_label_index[0], pos_edge_label_index[1]):
        pred_probs_pos_edge_label.append(adj_rec_probs[edge[0].item(), edge[1].item()].item())
    pred_probs_neg_edge_label = []
    for edge in zip(neg_edge_label_index[0], neg_edge_label_index[1]):
        pred_probs_neg_edge_label.append(adj_rec_probs[edge[0].item(), edge[1].item()].item())
    
    # Create vector with label-ordered predicted probabilities
    pred_probs = np.hstack([pred_probs_pos_edge_label, pred_probs_neg_edge_label])
    
    # Create vector with label-ordered ground truth labels
    labels = np.hstack([np.ones(len(pred_probs_pos_edge_label)), np.zeros(len(pred_probs_neg_edge_label))])

    # Calculate threshold independent metrics
    auroc_score = skm.roc_auc_score(labels, pred_probs)
    auprc_score = skm.average_precision_score(labels, pred_probs)

    if debug == True:
        print(f"Labels: {labels}", "\n")
        print(f"Predicted probabilities: {pred_probs}", "\n")
        
    # Get the optimal classification probability threshold above which an edge 
    # is classified as positive so that the threshold bestimizes the accuracy 
    # over the sampled (balanced) set of positive and negative edges.
    all_acc_score = {}
    best_acc_score = 0
    for threshold in np.arange(0.01, 1, 0.005):
        preds_labels = (pred_probs > threshold).astype("int")
        acc_score = skm.accuracy_score(labels, preds_labels)
        all_acc_score[threshold] = acc_score
        if acc_score > best_acc_score:
            best_acc_score = acc_score

    # Get the optimal classification probability threshold above which an edge 
    # is classified as positive so that the threshold bestimizes the f1 score 
    # over the sampled (balanced) set of positive and negative edges.
    all_f1_score = {}
    best_f1_score = 0
    for threshold in np.arange(0.01, 1, 0.005):
        preds_labels = (pred_probs > threshold).astype("int")
        f1_score = skm.f1_score(labels, preds_labels)
        all_f1_score[threshold] = f1_score
        if f1_score > best_f1_score:
            best_f1_score = f1_score
    
    return auroc_score, auprc_score, best_acc_score, best_f1_score


def plot_eval_metrics(eval_scores_dict):
    """
    Plot the evaluation metrics.
    """

    # Plot epochs as integers
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for metric_key, metric_scores in eval_scores_dict.items():
        plt.plot(metric_scores, label = metric_key)
    plt.title("Evaluation metrics validation dataset")
    plt.ylabel("metric score")
    plt.xlabel("epoch")
    plt.legend(loc = "lower right")
    plt.savefig("images/eval_metrics.png")


def reduce_edges_per_node(A_rec_logits,
                          optimal_threshold,
                          edges_per_node,
                          reduction):
    """
    Reduce the edges of the reconstruced edge probability adjacency matrix to
    best_edge_target per node.

    Parameters
    ----------
    A_rec_logits:
        Reconstructed adjacency matrix with logits.
    optimal_threshold:
        Optimal classification threshold as calculated with 
        get_optimal_cls_threshold_and_accuracy().
    edges_per_node:
        Target for edges per node.
    reduction:
        "soft": Keep edges that are among the top <edges_per_node> edges for one
        of the two edge nodes.
        "hard": Keep edges that are among the top <edges_per_node> edgesfor both
        of the two edge nodes.
    Returns
    ----------
    A_rec_new
        The new reconstructed adjacency matrix with reduced edge predictions.
    """
    # Calculate adjacency matrix with edge probabilities
    adj_rec_probs = torch.sigmoid(A_rec_logits)
    A_rec = copy.deepcopy(adj_rec_probs)
    A_rec = (A_rec>optimal_threshold).int()
    A_rec_tmp = copy.deepcopy(adj_rec_probs)
    for node in range(0, A_rec_tmp.shape[0]):
        tmp = A_rec_tmp[node,:]
        A_rec_tmp[node,:] = (A_rec_tmp[node,:] >= np.sort(tmp)[-edges_per_node]).int()
    A_rec_new = A_rec + A_rec_tmp
    # Mark edges that have been recreated and are among the top 2 node edges for
    # one of the nodes
    A_rec_new = (A_rec_new == 2).int()
    # Make adjacency matrix symmetric
    A_rec_new = A_rec_new + A_rec_new.T 
    # keep edges that are among the top 2 node edges for one of the nodes
    if reduction == "soft": # union
        A_rec_new = (A_rec_new != 0).int()
    # keep edges that are among the top 2 node edges for both of the nodes
    elif reduction == "hard": # intersection
        A_rec_new = (A_rec_new == 2).int()
    return A_rec_new