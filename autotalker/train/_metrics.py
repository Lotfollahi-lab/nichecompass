import copy

import numpy as np
import torch
import sklearn.metrics as skm


def get_eval_metrics(
        A_rec_probs: torch.Tensor,
        edges_pos: np.ndarray,
        edges_neg: np.ndarray,
        debug: bool = False):
    """
    Get the evaluation metrics for the (balanced) sample of positive and 
    negative edges.

    Parameters
    ----------
    A_rec_probs:
        Reconstructed adjacency matrix with edge probabilities.
    edges_pos:
        Numpy array containing node indices of positive edges.
    edges_neg:
        Numpy array containing node indices of negative edges.
    Returns
    ----------
    auroc_score:
        Area under the receiver operating characteristic curve.
    auprc_score:
        Area under the precision-recall curve.
    acc_score:
        Accuracy under optimal classification threshold.
    f1_score:
        F1 score under optimal classification threshold.
    """
    # Collect predictions for each label (positive vs negative edge) separately
    pred_probs_pos_labels = []
    for edge in edges_pos:
        pred_probs_pos_labels.append(A_rec_probs[edge[0], edge[1]].item())
    pred_probs_neg_labels = []
    for edge in edges_neg:
        pred_probs_neg_labels.append(A_rec_probs[edge[0], edge[1]].item())
    
    # Create vector with label-ordered predicted probabilities
    pred_probs_all_labels = np.hstack([pred_probs_pos_labels,
                                       pred_probs_neg_labels])
    
    # Create vector with label-ordered ground truth labels
    all_labels = np.hstack([np.ones(len(pred_probs_pos_labels)),
                            np.zeros(len(pred_probs_neg_labels))])
    
    auroc_score = skm.roc_auc_score(all_labels, pred_probs_all_labels)
    auprc_score = skm.average_precision_score(all_labels, pred_probs_all_labels)

    if debug == True:
        print(f"Labels: {all_labels}", "\n")
        print(f"Predicted probabilities: {pred_probs_all_labels}", "\n")

    # Get the optimal classification probability threshold above which an edge 
    # is classified as positive so that the threshold maximizes the accuracy 
    # over the sampled (balanced) set of positive and negative edges.
    all_acc_score = {}
    max_acc_score = 0
    optimal_threshold = 0
    for threshold in np.arange(0.01, 1, 0.005):
        preds_all_labels = (pred_probs_all_labels > threshold).astype("int")
        acc_score = skm.accuracy_score(all_labels, preds_all_labels)
        all_acc_score[threshold] = acc_score
        if acc_score > max_acc_score:
            max_acc_score = acc_score
            optimal_threshold = threshold
    preds_all_labels = (pred_probs_all_labels > optimal_threshold).astype("int")
    acc_score = skm.accuracy_score(all_labels, preds_all_labels)

    # Get the optimal classification probability threshold above which an edge 
    # is classified as positive so that the threshold maximizes the f1 score 
    # over the sampled (balanced) set of positive and negative edges.
    all_f1_score = {}
    max_f1_score = 0
    optimal_threshold = 0
    for threshold in np.arange(0.01, 1, 0.005):
        preds_all_labels = (pred_probs_all_labels > threshold).astype("int")
        f1_score = skm.f1_score(all_labels, preds_all_labels)
        all_f1_score[threshold] = f1_score
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            optimal_threshold = threshold
    preds_all_labels = (pred_probs_all_labels > optimal_threshold).astype("int")
    f1_score = skm.f1_score(all_labels, preds_all_labels)
    
    return auroc_score, auprc_score, acc_score, f1_score


def reduce_edges_per_node(A_rec_logits,
                          optimal_threshold,
                          edges_per_node,
                          reduction):
    """
    Reduce the edges of the reconstruced edge probability adjacency matrix to
    max_edge_target per node.

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
    A_rec_probs = torch.sigmoid(A_rec_logits)
    A_rec = copy.deepcopy(A_rec_probs)
    A_rec = (A_rec>optimal_threshold).int()
    A_rec_tmp = copy.deepcopy(A_rec_probs)
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