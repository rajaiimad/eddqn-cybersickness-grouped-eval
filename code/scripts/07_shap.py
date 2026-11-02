"""SHAP attribution of the E-DDQN DECISION MARGIN dQ = Q(s,alarm) - Q(s,silent).

For each outer fold: retrain the selected E-DDQN configuration on inner split 0
(seed 1000, identical to the first final model of that fold), compute SHAP
values of dQ for up to 400 test samples with a background of 200 training
samples (DeepExplainer with an additivity check; GradientExplainer, i.e.
expected gradients, when DeepExplainer does not support a layer; the method
used per fold is recorded in shap_method_by_fold.csv), and aggregate mean
|SHAP| per feature over folds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn import CONFIGS, RESULTS  # noqa: E402
from eddqn.data import build_datasets, fold_frames, prepare_fold, split_tables  # noqa: E402
from eddqn.experiment import make_model  # noqa: E402
from eddqn.features import display_name, feature_columns  # noqa: E402

torch.set_num_threads(2)
PROTOCOL = json.loads((CONFIGS / "protocol.json").read_text())
OUT = RESULTS / "tables"


class Margin(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def forward(self, x):
        out = self.q(x)
        return (out[:, 1] - out[:, 0]).unsqueeze(1)


if __name__ == "__main__":
    long, _, _ = build_datasets()
    tables = split_tables(long, PROTOCOL)
    sel = json.loads((RESULTS / "selected_configs.json").read_text())["E-DDQN"]
    rows, per_fold = [], []
    rng = np.random.default_rng(0)
    for k in range(PROTOCOL["k_outer"]):
        cfg = sel[str(k)]["config"]
        tr, val, test = fold_frames(long, tables, k, 0)
        fd = prepare_fold(tr, {"val": val, "test": test}, with_profile=True)
        names = feature_columns(fd.feature_names, cfg["feature_set"])
        cols = [fd.feature_names.index(c) for c in names]
        m = make_model("E-DDQN", cfg, len(cols), 1000)
        m.fit(fd.X_tr[:, cols], fd.y_tr, fd.meta_tr.session_id.to_numpy(), fd.X_ev["val"][:, cols], fd.y_ev["val"])
        net = Margin(m.online).eval()
        bg = torch.from_numpy(fd.X_tr[rng.choice(len(fd.X_tr), min(200, len(fd.X_tr)), replace=False)][:, cols])
        Xte = fd.X_ev["test"][:, cols]
        idx = rng.choice(len(Xte), min(400, len(Xte)), replace=False)
        xs = torch.from_numpy(Xte[idx])
        np.random.seed(1000 + k)   # GradientExplainer samples with np.random / torch
        torch.manual_seed(1000 + k)
        try:
            sv = shap.DeepExplainer(net, bg).shap_values(xs, check_additivity=True)
            method = "DeepExplainer (additivity checked)"
        except Exception as e:  # unsupported layer (e.g. LayerNorm) -> expected gradients
            sv = shap.GradientExplainer(net, bg).shap_values(xs)
            method = f"GradientExplainer (DeepExplainer failed: {type(e).__name__})"
        sv = np.asarray(sv).reshape(len(idx), len(cols))
        imp = np.abs(sv).mean(0)
        per_fold.append({"outer": k, "method": method, "feature_set": cfg["feature_set"], "n_features": len(cols)})
        for n, v in zip(names, imp):
            rows.append({"outer": k, "feature": n, "mean_abs_shap": float(v)})
        print(k, method, sorted(zip(imp, names))[-5:], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "shap_margin_by_fold.csv", index=False)
    agg = (df.groupby("feature").agg(mean_abs_shap=("mean_abs_shap", "mean"), folds_using=("outer", "nunique"))
           .sort_values("mean_abs_shap", ascending=False).reset_index())
    agg.to_csv(OUT / "shap_margin_summary.csv", index=False)
    pd.DataFrame(per_fold).to_csv(OUT / "shap_method_by_fold.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = agg.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh([display_name(f) for f in top.feature], top.mean_abs_shap, color="#3b6ea5")
    ax.set_xlabel(r"mean |SHAP| of $\Delta Q$ (averaged over outer folds)")
    ax.set_title("E-DDQN decision-margin attributions")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / "figures" / f"shap_margin.{ext}", dpi=200)
