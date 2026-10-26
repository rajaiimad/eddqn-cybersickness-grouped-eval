"""Cost-sensitive supervised baselines on identical inputs and splits.

All models are trained with the same 5:1 misclassification cost expressed as
a positive-class weight w = C_FN / C_FP = 5.  With such a weight the
population minimiser of the weighted log-loss is
    p_w = w p / (w p + 1 - p),
so p_w >= 0.5  <=>  p >= 1/6, the Bayes cut-off of the 5:1 cost.  ``score``
returns the implied un-weighted posterior p = p_w / (w - (w - 1) p_w), which
puts every model (and the Q-value-derived score of the agents) on the same
"implied P(y=1|s)" scale; operating thresholds are then chosen on validation
predictions only.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .agents import QNet
from .metrics import C_FN, C_FP, min_cost_value, ranking_metrics

W_POS = C_FN / C_FP


def unweight(p_w, w=W_POS):
    p_w = np.clip(p_w, 1e-7, 1 - 1e-7)
    return p_w / (w - (w - 1) * p_w)


class CostSensitiveMLP:
    """Same trunk family as the agents' Q-network, single-logit output,
    positive-class-weighted BCE (w = 5), checkpoint chosen on validation."""

    def __init__(self, hidden=(256, 512, 256), dropout=0.2, norm="batch", lr=1e-3, weight_decay=1e-4,
                 batch_size=512, epochs=60, eval_every=2, sampling="uniform", seed=0, w_pos=W_POS, **_):
        self.kw = dict(hidden=tuple(hidden), dropout=dropout, norm=norm)
        self.lr, self.wd, self.bs, self.epochs = lr, weight_decay, batch_size, epochs
        self.eval_every, self.sampling, self.seed, self.w_pos = eval_every, sampling, seed, w_pos

    def fit(self, X, y, sessions=None, X_val=None, y_val=None, log=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        net = QNet(X.shape[1], n_out=1, dueling=False, **self.kw)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.w_pos, dtype=torch.float32))
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
        best, best_state, self.best_epoch = (np.inf, -np.inf), copy.deepcopy(net.state_dict()), 0
        self.history = []
        n = len(X)
        for ep in range(1, self.epochs + 1):
            net.train()
            if self.sampling == "balanced":   # class-balanced minibatches (+ cost weight)
                i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
                perm = np.concatenate([rng.choice(i0, n // 2), rng.choice(i1, n - n // 2)])
                rng.shuffle(perm)
            else:
                perm = rng.permutation(n)
            for st in range(0, n - 1, self.bs):
                b = perm[st:st + self.bs]
                if len(b) < 2:
                    continue
                loss = lossf(net(Xt[b]).squeeze(1), yt[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            self.net = net
            rec = {"epoch": ep}
            if X_val is not None and (ep % self.eval_every == 0 or ep == self.epochs):
                sv = self.score(X_val)
                crit, prauc = min_cost_value(sv, y_val), ranking_metrics(sv, y_val)["pr_auc"]
                rec.update(val_min_cost=crit, val_pr_auc=prauc)
                if (-crit, prauc) > (-best[0], best[1]):
                    best, self.best_epoch = (crit, prauc), ep
                    best_state = copy.deepcopy(net.state_dict())
            self.history.append(rec)
        if X_val is not None:
            net.load_state_dict(best_state)
        self.net = net
        return self

    @torch.no_grad()
    def score(self, X):
        self.net.eval()
        p_w = torch.sigmoid(self.net(torch.from_numpy(X)).squeeze(1)).numpy()
        return unweight(p_w, self.w_pos)


class SkModel:
    """Wrapper for scikit-learn classifiers with the 5:1 class weight."""

    def __init__(self, kind, seed=0, **hp):
        self.kind, self.seed, self.hp = kind, seed, hp

    def _make(self):
        cw = {0: 1.0, 1: W_POS}
        if self.kind == "logreg":
            return LogisticRegression(C=self.hp.get("C", 1.0), class_weight=cw, max_iter=5000,
                                      solver="lbfgs")
        if self.kind == "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=self.hp.get("learning_rate", 0.05), max_iter=self.hp.get("max_iter", 200),
                max_leaf_nodes=self.hp.get("max_leaf_nodes", 31), min_samples_leaf=self.hp.get("min_samples_leaf", 20),
                l2_regularization=self.hp.get("l2_regularization", 0.0), class_weight=cw,
                early_stopping=False, random_state=self.seed)
        if self.kind == "rf":
            return RandomForestClassifier(
                n_estimators=self.hp.get("n_estimators", 300), max_depth=self.hp.get("max_depth", None),
                min_samples_leaf=self.hp.get("min_samples_leaf", 5), max_features=self.hp.get("max_features", "sqrt"),
                class_weight=cw, n_jobs=1, random_state=self.seed)
        raise ValueError(self.kind)

    def fit(self, X, y, sessions=None, X_val=None, y_val=None, log=None):
        self.m = self._make().fit(X, y)
        self.best_epoch = None
        self.history = []
        return self

    def score(self, X):
        return unweight(self.m.predict_proba(X)[:, 1])
