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

from deeplinc.train.metrics import eval_metrics

# 5 nodes, 2 dimensions latent space simulation
torch.manual_seed(0)
Z = torch.randn(5, 2)

edges_test_pos = np.array([[0, 3],
                           [1, 2],
                           [1, 4],
                           [2, 3]])
edges_test_neg = np.array([[0, 1],
                           [0, 2],
                           [0, 4],
                           [2, 4]])

def get_prob(Z, edges_test_pos, edges_test_neg):
    # if emb is None:
    #     feed_dict.update({placeholders['dropout']: 0})
    #     emb = sess.run(model.z_mean, feed_dict=feed_dict)
    def sigmoid(x):
        if x >= 0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
            return 1.0/(1+np.exp(-x))
        else:
            return np.exp(x)/(1+np.exp(x))
    # Predict on test set of edges
    adj_rec = np.dot(Z, Z.T)
    preds = []
    pos = []
    for e in edges_test_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
    preds_neg = []
    neg = []
    for e in edges_test_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
    labels_all = np.hstack((np.array(['connections between not_disrupted cells' for i in range(len(preds))]), np.array(['connections between disrupted cells' for i in range(len(preds))])))
    preds_all = np.hstack([preds, preds_neg])
    return np.hstack((labels_all.reshape(-1,1),preds_all.reshape(-1,1)))

test = get_prob(Z, edges_test_pos, edges_test_neg)
print(test)