"""Nested, grouped evaluation protocol shared by every model.

Outer loop  : K_OUTER grouped folds over the 21 linkage groups of the curated
              release.  Each outer test fold is touched exactly once, by
              the frozen protocol, after all choices for that fold are made.
Inner loop  : K_INNER grouped folds over the remaining (development) groups,
              used for (i) random-search hyper-parameter optimisation with an
              identical budget per model family, (ii) checkpoint selection
              (neural models) and (iii) choosing operating thresholds from
              pooled out-of-fold (OOF) validation scores.
Final model : for each seed, the K_INNER cross-fitted models trained with the
              selected configuration; the outer-test score is the mean of
              their scores ("cross-fitted ensemble").  The same recipe is used
              for every model, so comparisons are matched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .agents import AgentConfig, ValueAgent
from .data import fold_frames, prepare_fold, split_tables  # noqa: F401
from .features import feature_columns
from .metrics import (C_FN, C_FP, min_cost_value, ranking_metrics, threshold_for_recall,
                      threshold_for_specificity, threshold_min_cost)
from .supervised import CostSensitiveMLP, SkModel

# ---------------------------------------------------------------- families
AGENT_FIXED = {
    # proposed method: every E-DDQN component switched on
    "E-DDQN": dict(reward="asymmetric", dueling=True, double=True, target_update="soft",
                   replay="balanced"),
    # contextual-bandit special case (gamma = 0: no bootstrapping, so the
    # target network / double estimator are inactive by construction)
    "E-CB": dict(reward="asymmetric", dueling=True, double=True, target_update="soft",
                 replay="balanced", gamma=0.0),
    # standard DQN baseline (Mnih et al. 2015): symmetric +-1 reward, single
    # max-operator target, periodic hard target copies, uniform replay
    "DQN": dict(reward="symmetric", dueling=False, double=False, target_update="hard",
                replay="uniform", full_info=False),
}


# Post-hoc baselines added after the frozen primary analysis
# (configs/posthoc.json): an existing family restricted to a fixed feature set.
POSTHOC_BASE = {"Time-LR": "LogReg", "TimeGame-HGB": "HGB"}


def make_model(family: str, cfg: dict, n_in: int, seed: int):
    family = POSTHOC_BASE.get(family, family)
    if family in AGENT_FIXED or family.startswith("E-DDQN"):
        c = AgentConfig(**{k: (tuple(v) if k == "hidden" else v) for k, v in cfg.items()
                           if k in AgentConfig.__dataclass_fields__})
        c.seed = seed
        return ValueAgent(n_in, c)
    if family == "CS-MLP":
        return CostSensitiveMLP(seed=seed, **cfg)
    if family in ("LogReg", "HGB", "RF"):
        return SkModel({"LogReg": "logreg", "HGB": "hgb", "RF": "rf"}[family], seed=seed, **cfg)
    raise ValueError(family)


def sample_configs(space: dict, fixed: dict, n: int, seed: int) -> list[dict]:
    """Random search over a discrete space (identical budget per family)."""
    rng = np.random.default_rng(seed)
    if "__grid__" in space:
        grid = space["__grid__"]
        return [{**fixed, **g} for g in grid][:n]
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < 100 * n:
        tries += 1
        c = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = json.dumps(c, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append({**c, **fixed})
    return out


# ------------------------------------------------------------- one job
def run_job(job: dict, cache: dict) -> dict:
    """Train one model on one inner split; return validation (and, only for
    FINAL jobs, outer-test) scores.  HPO jobs never compute test scores."""
    torch.set_num_threads(1)
    fd = cache[(job["outer"], job["inner"])]
    cols = [fd.feature_names.index(c) for c in feature_columns(fd.feature_names, job["config"]["feature_set"])]
    Xtr, Xva = fd.X_tr[:, cols], fd.X_ev["val"][:, cols]
    t0 = time.time()
    m = make_model(job["family"], job["config"], Xtr.shape[1], job["seed"])
    m.fit(Xtr, fd.y_tr, fd.meta_tr["session_id"].to_numpy(), Xva, fd.y_ev["val"])
    out = {k: job[k] for k in ("family", "outer", "inner", "seed", "config_id", "stage")}
    sv = m.score(Xva)
    out["val_score"] = sv.astype(np.float32)
    out["val_min_cost"] = min_cost_value(sv, fd.y_ev["val"])
    out["val_pr_auc"] = ranking_metrics(sv, fd.y_ev["val"])["pr_auc"]
    out["best_epoch"] = getattr(m, "best_epoch", None)
    out["history"] = getattr(m, "history", [])
    if job["stage"] == "final":
        Xte = fd.X_ev["test"][:, cols]
        out["test_score"] = m.score(Xte).astype(np.float32)
        if hasattr(m, "q_values"):
            out["val_q"] = m.q_values(Xva).astype(np.float32)
            out["test_q"] = m.q_values(Xte).astype(np.float32)
        att = cache.get(("attacks", job["outer"], job["inner"]))
        if att is not None and job.get("score_attacks", True):
            out["attack_scores"] = {name: m.score(X[:, cols]).astype(np.float32) for name, X in att.items()}
    out["seconds"] = time.time() - t0
    return out


def build_cache(long, tables, protocol, include_test=False, outers=None):
    """Fold-wise preprocessing (fitted on the inner-training rows only).
    HPO caches are built WITHOUT the outer-test rows, so test features are
    not even computed until the final stage."""
    cache = {}
    for k in (outers if outers is not None else range(protocol["k_outer"])):
        for j in range(protocol["k_inner"]):
            tr, val, test = fold_frames(long, tables, k, j)
            ev = {"val": val}
            if include_test:
                ev["test"] = test
            cache[(k, j)] = prepare_fold(tr, ev, with_profile=True)
    return cache


# ----------------------------------------------------- operating points
def operating_thresholds(oof_score, oof_y, protocol) -> dict:
    th = {"D1_min_cost": threshold_min_cost(oof_score, oof_y),
          "D0_bayes_1_6": C_FP / (C_FP + C_FN)}
    for r in protocol["matched_recalls"]:
        th[f"recall>={r}"] = threshold_for_recall(oof_score, oof_y, r)
    for s in protocol["matched_specificities"]:
        th[f"spec>={s}"] = threshold_for_specificity(oof_score, oof_y, s)
    return th


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
