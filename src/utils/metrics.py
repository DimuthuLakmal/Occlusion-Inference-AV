import numpy as np
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def _counts(outputs_sig, targets, mask, threshold=0.5):
    probs = (outputs_sig * mask).flatten()
    labels = (targets * mask).flatten()

    preds = (probs >= threshold).float()
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    tn = (((1 - preds) * (1 - labels))).sum()

    tp, fp, fn, tn = [x.item() for x in (tp, fp, fn, tn)]
    return tp, fp, fn, tn, probs, labels, mask.flatten().bool()


def _ratios_from_counts(tp, fp, fn, tn):
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),  # TPR
        "acc_free": float(specificity),  # TNR (negative accuracy)
        "f1": float(f1),
    }


def compute_metrics(outputs_sig, targets, mask, threshold=0.5):
    # outputs_sig, targets, mask are same shape; values in {0,1} for targets/mask
    tp, fp, fn, tn, probs, labels, m = _counts(outputs_sig, targets, mask, threshold)
    result = _ratios_from_counts(tp, fp, fn, tn)

    roc_auc = None
    if _HAS_SK:
        y_true = labels[m].detach().cpu().numpy()
        y_prob = probs[m].detach().cpu().numpy()
        if y_true.size > 0 and len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))

    result["roc_auc"] = roc_auc
    return result


def accumulate_counts(outputs_sig, targets, mask, threshold=0.5):
    """Per-batch raw tp/fp/fn/tn (+ pooled probs/labels) for later aggregation.

    Callers should sum these across all batches and call
    compute_metrics_from_counts() once on the totals. Averaging per-batch
    ratios (as compute_metrics() returns) instead is NOT invariant to
    batch size: it's a macro-average over batches rather than the correct
    micro-average over samples, and gets worse when positives are sparse
    or the last batch is a partial one.
    """
    tp, fp, fn, tn, probs, labels, m = _counts(outputs_sig, targets, mask, threshold)
    y_true = labels[m].detach().cpu().numpy()
    y_prob = probs[m].detach().cpu().numpy()
    return tp, fp, fn, tn, y_true, y_prob


def compute_metrics_from_counts(tp, fp, fn, tn, y_true=None, y_prob=None):
    result = _ratios_from_counts(tp, fp, fn, tn)

    roc_auc = None
    if _HAS_SK and y_true is not None and y_true.size > 0 and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))

    result["roc_auc"] = roc_auc
    return result
