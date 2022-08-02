import numpy as np
import torch
import copy
from scipy.special import expit
from sklearn import metrics
from sklearn.metrics  import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


class LinkPredictionMetrics():
    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def get_roc_score(self, emb, feas):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            if x >= 0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
                return 1.0/(1+np.exp(-x))
            else:
                return np.exp(x)/(1+np.exp(x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in self.edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        acc_score = accuracy_score(labels_all, np.round(preds_all))

        return roc_score, ap_score, acc_score, emb

    def get_prob(self, emb, feas):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            if x >= 0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
                return 1.0/(1+np.exp(-x))
            else:
                return np.exp(x)/(1+np.exp(x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in self.edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

        labels_all = np.hstack((np.array(['connections between not_disrupted cells' for i in range(len(preds))]), np.array(['connections between disrupted cells' for i in range(len(preds))])))
        preds_all = np.hstack([preds, preds_neg])

        return np.hstack((labels_all.reshape(-1,1),preds_all.reshape(-1,1)))


def get_optimal_cls_threshold_and_accuracy(Z, edges_test_pos, edges_test_neg):
    """
    Select the classification threshold that maximizes the accuracy for the
    prediction of positive and negative test edges and calculate the accuracy.

    Parameters
    ----------
    Z:
        Latent space features that are fed into the decoder.
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
    A_rec_probs = torch.sigmoid(torch.mm(Z, Z.T))

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



import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score