import copy
import numpy as np
import torch
from sklearn.metrics  import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


def get_eval_metrics(A_rec_logits, edges_test_pos, edges_test_neg, acc_threshold: float = 0.5):
    """
    Get the evaluation metrics
    prediction of positive and negative test edges and calculate the accuracy.

    Parameters
    ----------
    A_rec_logits:
        Reconstructed adjacency matrix with logits.
    edges_test_pos:
        Numpy array containing node indices of positive edges.
    edges_test_neg:
        Numpy array containing node indices of negative edges.
    acc_threshold:
        Threshold to be used for the calculation of the accuracy.
    Returns
    ----------
    optimal_cls_threshold
        Classification threshold that maximizes accuracy.
    max_acc_score
        Accuracy under optimal classification threshold. 
    """
    # Calculate adjacency matrix with edge probabilities
    A_rec_probs = torch.sigmoid(A_rec_logits)
    # Collect predictions for each label (positive vs negative edge) separately
    pred_probs_pos_labels = []
    for edge in edges_test_pos:
        pred_probs_pos_labels.append(A_rec_probs[edge[0], edge[1]])
    pred_probs_neg_labels = []
    for edge in edges_test_neg:
        pred_probs_neg_labels.append(A_rec_probs[edge[0], edge[1]])
    
    # Create tensor of ground truth labels
    pred_probs_all_labels = torch.hstack([torch.tensor(pred_probs_pos_labels),
                                          torch.tensor(pred_probs_neg_labels)])
    preds_all_labels = (pred_probs_all_labels>acc_threshold).int()
    all_labels = torch.hstack([torch.ones(len(pred_probs_pos_labels)),
                               torch.zeros(len(pred_probs_neg_labels))])
    
    roc_score = roc_auc_score(all_labels, pred_probs_all_labels)
    ap_score = average_precision_score(all_labels, pred_probs_all_labels)
    acc_score = accuracy_score(all_labels, preds_all_labels)
    
    return roc_score, ap_score, acc_score


def get_optimal_cls_threshold_and_accuracy(A_rec_logits, edges_test_pos, edges_test_neg):
    """
    Select the classification threshold that maximizes the accuracy for the
    prediction of positive and negative test edges and calculate the accuracy.

    Parameters
    ----------
    A_rec_logits:
        Reconstructed adjacency matrix with logits.
    edges_test_pos:
        Numpy array containing node indices of positive edges.
    edges_test_neg:
        Numpy array containing node indices of negative edges.
    Returns
    ----------
    optimal_cls_threshold
        Classification threshold that maximizes accuracy.
    max_acc_score
        Accuracy under optimal classification threshold. 
    """
    # Calculate adjacency matrix with edge probabilities
    A_rec_probs = torch.sigmoid(A_rec_logits)
    # Collect predicted probabilities for positive and negative edges separately
    pred_probs_pos_labels = []
    for edge in edges_test_pos:
        pred_probs_pos_labels.append(A_rec_probs[edge[0], edge[1]])
    pred_probs_neg_labels = []
    for edge in edges_test_neg:
        pred_probs_neg_labels.append(A_rec_probs[edge[0], edge[1]])
    
    all_labels = torch.hstack([torch.ones(len(pred_probs_pos_labels)),
    
                               torch.zeros(len(pred_probs_neg_labels))])
    
    # Calculate accuracies for all thresholds and store best accuracy and 
    # threshold
    all_acc_score = {}
    max_acc_score = 0
    optimal_threshold = 0
    for threshold in np.arange(0.01, 1, 0.005):
        pred_probs_all_labels = torch.hstack(
            [torch.tensor(pred_probs_pos_labels),
             torch.tensor(pred_probs_neg_labels)])
        preds_all_labels = (pred_probs_all_labels>threshold).int()
        acc_score = accuracy_score(all_labels, preds_all_labels)
        all_acc_score[threshold] = acc_score
        if acc_score > max_acc_score:
            max_acc_score = acc_score
            optimal_threshold = threshold

    return optimal_threshold, max_acc_score


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