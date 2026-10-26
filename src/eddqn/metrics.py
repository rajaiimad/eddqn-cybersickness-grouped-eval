"""Metrics, operating-point selection and paired statistics.

All reported numbers are computed from saved (score, label) arrays by the
functions in this module, so every table can be regenerated from
``results/predictions``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

C_FN, C_FP = 5.0, 1.0           # pre-registered 5:1 safety cost structure
BAYES_THRESHOLD = C_FP / (C_FP + C_FN)   # = 1/6: cost-optimal cut-off on a calibrated P(y=1|s)


def confusion(pred, y):
    pred = np.asarray(pred).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def binary_metrics(pred, y) -> dict:
    tp, tn, fp, fn = confusion(pred, y)
    n = tp + tn + fp + fn
    rec, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    prec, npv = _div(tp, tp + fp), _div(tn, tn + fn)
    f1 = _div(2 * prec * rec, prec + rec) if not math.isnan(prec) else float("nan")
    f2 = _div(5 * prec * rec, 4 * prec + rec) if not math.isnan(prec) else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _div(tp * tn - fp * fn, den) if den else 0.0
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "N": n,
        "accuracy": _div(tp + tn, n), "precision": prec, "recall": rec,
        "specificity": spec, "npv": npv, "f1": f1, "f2": f2,
        "balanced_accuracy": (rec + spec) / 2, "mcc": mcc,
        "g_mean": math.sqrt(rec * spec) if not (math.isnan(rec) or math.isnan(spec)) else float("nan"),
        "fpr": 1 - spec, "fnr": 1 - rec,
        "expected_cost": (C_FN * fn + C_FP * fp) / n,       # per sample, 5:1
        "alarm_rate": _div(tp + fp, n),
    }


def ranking_metrics(score, y) -> dict:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "prevalence": float(np.mean(y))}


def all_metrics(score, y, threshold) -> dict:
    pred = (np.asarray(score) >= threshold).astype(int)
    m = binary_metrics(pred, y)
    m.update(ranking_metrics(score, y))
    m["threshold"] = float(threshold)
    return m


# ------------------------------------------------------------ thresholds
def _candidate_thresholds(score):
    s = np.unique(np.asarray(score, dtype=float))
    mids = (s[1:] + s[:-1]) / 2 if len(s) > 1 else s
    return np.concatenate([[s.min() - 1e-9], mids, [s.max() + 1e-9]])


def sweep(score, y):
    """Vectorised confusion counts for every candidate threshold (pred = score >= t)."""
    score = np.asarray(score, float)
    y = np.asarray(y, int)
    t = _candidate_thresholds(score)
    order = np.argsort(-score)
    s_sorted, y_sorted = score[order], y[order]
    # number of samples with score >= t
    k = np.searchsorted(-s_sorted, -t, side="right")
    ctp = np.concatenate([[0], np.cumsum(y_sorted)])
    cfp = np.concatenate([[0], np.cumsum(1 - y_sorted)])
    tp, fp = ctp[k], cfp[k]
    P, N = y.sum(), len(y) - y.sum()
    fn, tn = P - tp, N - fp
    return t, tp, fp, fn, tn


def threshold_min_cost(score, y, c_fn=C_FN, c_fp=C_FP) -> float:
    """Validation rule D1: threshold minimising expected 5:1 cost.
    Ties are broken towards the threshold closest to the Bayes cut-off 1/6."""
    t, tp, fp, fn, tn = sweep(score, y)
    cost = c_fn * fn + c_fp * fp
    best = np.flatnonzero(cost == cost.min())
    return float(t[best[np.argmin(np.abs(t[best] - BAYES_THRESHOLD))]])


def threshold_for_recall(score, y, target: float) -> float:
    """Validation rule D2: the largest threshold whose validation recall >= target."""
    t, tp, fp, fn, tn = sweep(score, y)
    rec = tp / max(tp[-1] + fn[-1], 1)
    ok = np.flatnonzero(rec >= target)
    return float(t[ok].max()) if len(ok) else float(t.min())


def threshold_for_specificity(score, y, target: float) -> float:
    t, tp, fp, fn, tn = sweep(score, y)
    spec = tn / np.maximum(tn + fp, 1)
    ok = np.flatnonzero(spec >= target)
    return float(t[ok].min()) if len(ok) else float(t.max())


def min_cost_value(score, y) -> float:
    t, tp, fp, fn, tn = sweep(score, y)
    return float((C_FN * fn + C_FP * fp).min() / len(y))


# ------------------------------------------------------------ statistics
def mcnemar(pred_a, pred_b, y) -> dict:
    """Paired McNemar test on the SAME test samples.

    b = A correct & B wrong, c = A wrong & B correct.  Reports the exact
    binomial p-value and the continuity-corrected chi-square.  Consistency
    check: b - c must equal errors(B) - errors(A).
    """
    y = np.asarray(y)
    ca = np.asarray(pred_a) == y
    cb = np.asarray(pred_b) == y
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    both_wrong = int((~ca & ~cb).sum())
    both_right = int((ca & cb).sum())
    n = b + c
    p_exact = float(stats.binomtest(min(b, c), n, 0.5).pvalue) if n else 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n if n else 0.0
    p_chi2 = float(stats.chi2.sf(chi2, 1)) if n else 1.0
    err_a, err_b = int((~ca).sum()), int((~cb).sum())
    assert b - c == err_b - err_a, "McNemar table inconsistent with error counts"
    return {"b_A_right_B_wrong": b, "c_A_wrong_B_right": c, "both_right": both_right,
            "both_wrong": both_wrong, "errors_A": err_a, "errors_B": err_b,
            "chi2_cc": chi2, "p_chi2_cc": p_chi2, "p_exact": p_exact}


def cluster_bootstrap(stat_fn, groups, n_boot=2000, seed=0):
    """Percentile CI by resampling independent units (linkage groups) with
    replacement.  ``stat_fn(idx)`` returns a scalar for row indices ``idx``."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[g] for g in pick])
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    return {"estimate": float(stat_fn(np.arange(len(groups)))),
            "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5)),
            "n_boot_valid": int(len(vals))}


def paired_cluster_bootstrap_diff(metric_fn, pred_a, pred_b, y, groups, n_boot=2000, seed=0):
    """CI and two-sided bootstrap p-value for metric(A) - metric(B), resampling
    linkage groups (accounts for the strong within-session autocorrelation
    that the sample-level McNemar test ignores)."""
    pred_a, pred_b, y = map(np.asarray, (pred_a, pred_b, y))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    obs = metric_fn(pred_a, y) - metric_fn(pred_b, y)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[g] for g in pick])
        d = metric_fn(pred_a[idx], y[idx]) - metric_fn(pred_b[idx], y[idx])
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(obs), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p_boot_two_sided": float(min(1.0, p))}


def cost_of(pred, y):
    tp, tn, fp, fn = confusion(pred, y)
    return (C_FN * fn + C_FP * fp) / max(len(y), 1)
