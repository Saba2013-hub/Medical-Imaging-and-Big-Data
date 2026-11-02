
"""Metrics shared by all models. prob: (n, 5) class probabilities; y: grades 0-4."""
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, roc_auc_score, confusion_matrix

def referable_score(prob):
    return prob[:, 2:].sum(axis=1)                       # P(grade >= 2)

def threshold_for_sensitivity(y, prob, target=0.90):
    """Highest threshold on P(referable) whose sensitivity is >= target (choose on VALIDATION only)."""
    s, ref = referable_score(prob), (np.asarray(y) >= 2).astype(int)
    for t in np.sort(np.unique(s))[::-1]:
        if ((s >= t) & (ref == 1)).sum() / max(ref.sum(), 1) >= target:
            return float(t)
    return 0.0

def compute_metrics(y, prob, ref_threshold):
    y = np.asarray(y); pred = prob.argmax(axis=1)
    ref, s = (y >= 2).astype(int), referable_score(prob)
    ref_pred = (s >= ref_threshold).astype(int)
    tp, fn = ((ref_pred == 1) & (ref == 1)).sum(), ((ref_pred == 0) & (ref == 1)).sum()
    tn, fp = ((ref_pred == 0) & (ref == 0)).sum(), ((ref_pred == 1) & (ref == 0)).sum()
    out = {'QWK': cohen_kappa_score(y, pred, weights='quadratic'),
           'macro_F1': f1_score(y, pred, average='macro'),
           'accuracy': (pred == y).mean(),
           'referable_AUROC': roc_auc_score(ref, s),
           'referable_sensitivity': tp / max(tp + fn, 1),
           'referable_specificity': tn / max(tn + fp, 1)}
    for g, r in enumerate(recall_score(y, pred, labels=range(5), average=None, zero_division=0)):
        out[f'recall_grade_{g}'] = r
    return out
