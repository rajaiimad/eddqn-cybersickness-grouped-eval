"""Threshold-scale sensitivity of rule D1 (post hoc; configs/posthoc.json
'threshold_scale_sensitivity', committed before this script was run).

The D1 threshold of each (family, outer fold, seed) is chosen on pooled OOF
scores, each produced by ONE inner model, but applied to the outer-test score,
the MEAN of the three inner models' scores.  Here the unchanged threshold is
also applied to each inner model's own test score (the scale on which it was
chosen).  No model is trained and no threshold is re-chosen.

Out: results/tables/threshold_scale/{threshold_scale.csv, ts_macros.tex, ts_macros.json}
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn import CONFIGS, RESULTS  # noqa: E402
from eddqn.data import build_datasets, fold_frames, split_tables  # noqa: E402
from eddqn.experiment import operating_thresholds  # noqa: E402

PROTOCOL = json.loads((CONFIGS / "protocol.json").read_text())
KO, KI, SEEDS = PROTOCOL["k_outer"], PROTOCOL["k_inner"], PROTOCOL["final_seeds"]
FAMS = json.loads((CONFIGS / "posthoc.json").read_text())["threshold_scale_sensitivity"]["primary_models"]
OUT = RESULTS / "tables" / "threshold_scale"
KEY = {"E-DDQN": "EDDQN", "E-CB": "ECB", "DQN": "DQN", "CS-MLP": "CSMLP", "LogReg": "LR", "HGB": "HGB", "RF": "RF"}


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def rates(pred, y):
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {"cost": (5 * fn + fp) / len(y), "recall": 100 * (1 - fn / (y == 1).sum()), "fpr": 100 * fp / (y == 0).sum()}


def main():
    long, _, _ = build_datasets()
    tables = split_tables(long, PROTOCOL)
    val_y = {(k, j): fold_frames(long, tables, k, j)[1]["y"].to_numpy() for k in range(KO) for j in range(KI)}
    test_y = {k: fold_frames(long, tables, k)[1]["y"].to_numpy() for k in range(KO)}
    y_all = np.concatenate([test_y[k] for k in range(KO)])
    rows, spread = [], []
    for fam in FAMS:
        for s in SEEDS:
            pipe, single = [], {j: [] for j in range(KI)}
            for k in range(KO):
                parts = [load(RESULTS / "runs" / "final" / fam / f"o{k}_s{s}_i{j}.pkl") for j in range(KI)]
                oof = np.concatenate([p["val_score"] for p in parts])
                thr = operating_thresholds(oof, np.concatenate([val_y[(k, j)] for j in range(KI)]), PROTOCOL)["D1_min_cost"]
                te = np.stack([p["test_score"] for p in parts])          # 3 x n_test
                pipe.append((te.mean(0) >= thr).astype(int))
                for j in range(KI):
                    single[j].append((te[j] >= thr).astype(int))
                sd_single = float(np.mean([te[j].std() for j in range(KI)]))
                spread.append({"family": fam, "seed": s, "outer": k,
                               "ratio": float(te.mean(0).std() / sd_single) if sd_single > 0 else np.nan})
            r_pipe = rates(np.concatenate(pipe), y_all)
            r_single = [rates(np.concatenate(single[j]), y_all) for j in range(KI)]
            rows.append({"family": fam, "seed": s, **{f"pipe_{m}": v for m, v in r_pipe.items()},
                         **{f"single_{m}": float(np.mean([r[m] for r in r_single])) for m in r_pipe}})
    per_seed = pd.DataFrame(rows)
    summ = per_seed.groupby("family", sort=False).mean(numeric_only=True).drop(columns="seed")
    for m in ("cost", "recall", "fpr"):
        summ[f"delta_{m}"] = summ[f"single_{m}"] - summ[f"pipe_{m}"]
    sp = pd.DataFrame(spread).groupby("family", sort=False)["ratio"].agg(["median", "min", "max"])
    summ = summ.join(sp.add_prefix("spread_"))
    OUT.mkdir(parents=True, exist_ok=True)
    summ.round(4).to_csv(OUT / "threshold_scale.csv")
    per_seed.round(4).to_csv(OUT / "threshold_scale_per_seed.csv", index=False)

    # check: the recomputed pipeline reproduces the published seed-mean D1 results
    pub = json.loads((RESULTS / "tables" / "manuscript_macros.json").read_text())
    for fam in FAMS:
        assert abs(summ.loc[fam, "pipe_cost"] - float(pub[f"{KEY[fam]}CostCostMean"])) < 6e-4, fam

    ex = summ.drop(index="E-DDQN")
    rank_same = list(summ.pipe_cost.sort_values().index) == list(summ.single_cost.sort_values().index)
    f2 = lambda v: f"{v:+.3f}".replace("-", "$-$")
    f1 = lambda v: f"{v:+.1f}".replace("-", "$-$")
    mac = {
        "TSEDDQNCostPipe": f"{summ.loc['E-DDQN', 'pipe_cost']:.3f}", "TSEDDQNCostSingle": f"{summ.loc['E-DDQN', 'single_cost']:.3f}",
        "TSEDDQNDeltaCost": f2(summ.loc["E-DDQN", "delta_cost"]),
        "TSEDDQNDeltaRecall": f1(summ.loc["E-DDQN", "delta_recall"]), "TSEDDQNDeltaFPR": f1(summ.loc["E-DDQN", "delta_fpr"]),
        "TSOtherDeltaCostMin": f2(ex.delta_cost.min()), "TSOtherDeltaCostMax": f2(ex.delta_cost.max()),
        "TSAbsDeltaCostMax": f"{summ.delta_cost.abs().max():.3f}",
        "TSMaxAbsDeltaFamily": summ.delta_cost.abs().idxmax(),
        "TSSpreadEDDQN": f"{summ.loc['E-DDQN', 'spread_median']:.2f}",
        "TSSpreadOtherMin": f"{ex.spread_median.min():.2f}", "TSSpreadOtherMax": f"{ex.spread_median.max():.2f}",
        "TSNeuralDeltaCostMin": f2(summ.loc[["E-DDQN", "E-CB", "DQN", "CS-MLP"], "delta_cost"].min()),
        "TSNeuralDeltaCostMax": f2(summ.loc[["E-DDQN", "E-CB", "DQN", "CS-MLP"], "delta_cost"].max()),
        "TSTreeLRDeltaCostMin": f2(summ.loc[["LogReg", "HGB", "RF"], "delta_cost"].min()),
        "TSTreeLRDeltaCostMax": f2(summ.loc[["LogReg", "HGB", "RF"], "delta_cost"].max()),
        "TSRankUnchanged": "unchanged" if rank_same else "changed",
    }
    (OUT / "ts_macros.json").write_text(json.dumps(mac, indent=1))
    (OUT / "ts_macros.tex").write_text("% generated by scripts/16_threshold_scale.py -- do not edit\n"
                                       + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    # supplementary table
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Model & \multicolumn{2}{c}{5:1 cost} & $\Delta$cost & $\Delta$recall & $\Delta$FPR & \multicolumn{2}{c}{spread ratio} \\",
             r" & reported & single model & & (pp) & (pp) & median & range \\", r"\midrule"]
    for fam, r in summ.iterrows():
        lines.append(f"{fam} & {r.pipe_cost:.3f} & {r.single_cost:.3f} & {f2(r.delta_cost)} & {f1(r.delta_recall)} & "
                     f"{f1(r.delta_fpr)} & {r.spread_median:.2f} & {r.spread_min:.2f}--{r.spread_max:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_ts.tex").write_text("\n".join(lines) + "\n")
    print(summ.round(3).to_string())
    print(json.dumps(mac, indent=1))


if __name__ == "__main__":
    main()
