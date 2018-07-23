import numpy as np
from sklearn.metrics import roc_auc_score


def accuracy(y_pred, y_true, thresh=0.5, reverse=False):
    y = (y_pred >= thresh) if not reverse else (y_pred <= thresh)
    return np.mean(y == y_true)

def auc(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)

def find_clf_threshold(y_pred, y_true, increment=0.01, reverse=False):
    thresh = 0
    best_acc = 0
    score_sorted = sorted(y_pred)
    for t in range(len(score_sorted)):
        acc = accuracy(y_pred, y_true, thresh=t, reverse=reverse)
        if acc > best_acc:
            thresh = t
            best_acc = acc
    return thresh, best_acc