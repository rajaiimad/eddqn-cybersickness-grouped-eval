"""Calibrate training budgets on the DEVELOPMENT-ONLY Oct-2019 sessions.

Nothing here touches the curated evaluation sessions.  Outputs timing and
validation-cost-by-epoch curves used to fix `fixed_training` and the HPO
budget in configs/protocol.json *before* the frozen run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn import RESULTS  # noqa: E402
from eddqn.data import build_datasets, grouped_folds, prepare_fold  # noqa: E402
from eddqn.experiment import AGENT_FIXED, make_model  # noqa: E402
from eddqn.features import feature_columns  # noqa: E402
from eddqn.metrics import min_cost_value, ranking_metrics  # noqa: E402

torch.set_num_threads(1)
OUT = RESULTS / "dev_calibration"
OUT.mkdir(parents=True, exist_ok=True)

_, dv, _ = build_datasets()
folds = grouped_folds(dv, 3, 99)
splits = []
for j in range(3):
    va = dv[dv.group_id.isin(folds.loc[folds.fold == j, "group_id"])]
    tr = dv[~dv.group_id.isin(va.group_id)]
    splits.append(prepare_fold(tr, {"val": va}, with_profile=True))

base_agent = dict(epochs=60, eval_every=2, chunk=256, learn_every=2, consolidation=5, eps_start=1.0,
                  eps_end=0.01, buffer=50000, grad_clip=1.0, eps_decay=0.95, warmup=4000, updates_per_learn=2,
                  lr=1e-3, weight_decay=1e-4, batch_size=512, dropout=0.2, norm="batch")
trials = [
    ("E-DDQN", {**base_agent, **AGENT_FIXED["E-DDQN"], "gamma": 0.9, "tau": 0.01, "rho": 0.5,
                "full_info": False, "hidden": [256, 512, 256], "feature_set": "telemetry+profile"}),
    ("E-DDQN", {**base_agent, **AGENT_FIXED["E-DDQN"], "gamma": 0.0, "tau": 0.01, "rho": 0.5,
                "full_info": True, "hidden": [64, 64], "feature_set": "exposure+profile"}),
    ("DQN", {**base_agent, **AGENT_FIXED["DQN"], "gamma": 0.9, "hard_every": 100, "hidden": [256, 512, 256],
             "feature_set": "telemetry+profile"}),
    ("CS-MLP", {"epochs": 60, "eval_every": 2, "hidden": [256, 512, 256], "dropout": 0.2, "norm": "batch",
                "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 512, "sampling": "uniform",
                "feature_set": "telemetry+profile"}),
]
rows, curves = [], []
for ti, (fam, cfg) in enumerate(trials):
    for j, fd in enumerate(splits):
        cols = [fd.feature_names.index(c) for c in feature_columns(fd.feature_names, cfg["feature_set"])]
        t0 = time.time()
        m = make_model(fam, cfg, len(cols), seed=0)
        m.fit(fd.X_tr[:, cols], fd.y_tr, fd.meta_tr.session_id.to_numpy(), fd.X_ev["val"][:, cols], fd.y_ev["val"])
        sec = time.time() - t0
        s = m.score(fd.X_ev["val"][:, cols])
        rows.append({"trial": ti, "family": fam, "inner": j, "seconds": sec, "best_epoch": m.best_epoch,
                     "val_min_cost": min_cost_value(s, fd.y_ev["val"]), **ranking_metrics(s, fd.y_ev["val"]),
                     "always_alarm_cost": float(1 - fd.y_ev["val"].mean())})
        for h in m.history:
            if "val_min_cost" in h:
                curves.append({"trial": ti, "family": fam, "inner": j, **h})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
pd.DataFrame(curves).to_csv(OUT / "curves.csv", index=False)
(OUT / "trials.json").write_text(json.dumps([{"family": f, "config": c} for f, c in trials], indent=1))
