import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import torch
from matplotlib.ticker import MaxNLocator


def get_eval_metrics(
        adj_rec_probs: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_labels: torch.Tensor):
    """
    Get the evaluation metrics for a (balanced) sample of positive and negative 
    edges.

    Parameters
    ----------
    adj_rec_probs:
        Tensor containing reconstructed adjacency matrix with edge 
        probabilities.
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
    edge_labels = edge_labels.detach().cpu().numpy()
    edge_label_index = edge_label_index.detach().cpu().numpy()

    # Collect predictions for each label (positive vs negative edge) separately
    pred_probs = np.array([])
    for edge in zip(edge_label_index[0], edge_label_index[1]):
        pred_probs = np.append(
            pred_probs,
            adj_rec_probs[edge[0], edge[1]].item())   

    # Calculate threshold independent metrics
    auroc_score = skm.roc_auc_score(edge_labels, pred_probs)
    auprc_score = skm.average_precision_score(edge_labels, pred_probs)
        
    # Get the optimal classification probability threshold above which an edge 
    # is classified as positive so that the threshold bestimizes the accuracy 
    # over the sampled (balanced) set of positive and negative edges.
    all_acc_score = {}
    best_acc_score = 0
    for threshold in np.arange(0.01, 1, 0.005):
        preds_labels = (pred_probs > threshold).astype("int")
        acc_score = skm.accuracy_score(edge_labels, preds_labels)
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
        f1_score = skm.f1_score(edge_labels, preds_labels)
        all_f1_score[threshold] = f1_score
        if f1_score > best_f1_score:
            best_f1_score = f1_score
    
    return auroc_score, auprc_score, best_acc_score, best_f1_score
    

def get_eval_metrics_old(
        adj_rec_probs: torch.Tensor,
        pos_edge_label_index: torch.Tensor,
        neg_edge_label_index: torch.Tensor):
    """
    Get the evaluation metrics for a (balanced) sample of positive and negative 
    edges.

    Parameters
    ----------
    adj_rec_probs:
        Tensor containing reconstructed adjacency matrix with edge 
        probabilities.
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
        pred_probs_pos_edge_label.append(
            adj_rec_probs[edge[0].item(), edge[1].item()].item())
    pred_probs_neg_edge_label = []
    for edge in zip(neg_edge_label_index[0], neg_edge_label_index[1]):
        pred_probs_neg_edge_label.append(
            adj_rec_probs[edge[0].item(), edge[1].item()].item())
    
    # Create vector with label-ordered predicted probabilities
    pred_probs = np.hstack([pred_probs_pos_edge_label,
                            pred_probs_neg_edge_label])
    
    # Create vector with label-ordered ground truth labels
    labels = np.hstack([np.ones(len(pred_probs_pos_edge_label)), 
                        np.zeros(len(pred_probs_neg_edge_label))])

    # Calculate threshold independent metrics
    auroc_score = skm.roc_auc_score(labels, pred_probs)
    auprc_score = skm.average_precision_score(labels, pred_probs)
        
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
    Plot evaluation metrics.

    Parameters
    ----------
    eval_scores_dict:
        Dictionary containing the eval metric scores to be plotted.
    """

    # Plot epochs as integers
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot eval metrics
    for metric_key, metric_scores in eval_scores_dict.items():
        plt.plot(metric_scores, label = metric_key)
    plt.title("Evaluation metrics validation dataset")
    plt.ylabel("metric score")
    plt.xlabel("epoch")
    plt.legend(loc = "lower right")

    # Retrieve figure
    fig = plt.gcf()
    return fig