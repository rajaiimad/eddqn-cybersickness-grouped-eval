"""Effective position of the untuned cut-off p_hat >= 1/6 (rule D0) on P(y=1|s).

Descriptive and post hoc (added in the pre-submission cross-check; recorded in
configs/posthoc.json '_later_additions').  No model is trained: the computation
uses only the selected configurations and the class prevalence of each inner
training split.

* Class-balanced replay (E-DDQN, E-CB: a fraction rho of non-events in every
  minibatch) and class-balanced minibatches of the CS-MLP (50/50) train on a
  class mix other than the prevalence pi of the training rows.  The learned
  scores then estimate the posterior under that mix, whose odds are k times the
  odds of P(y=1|s), with k = ((1-rho)/rho) * ((1-pi)/pi)  (k = (1-pi)/pi for
  50/50 minibatches).  A cut-off of 1/6 on the score is therefore a cut-off of
  1/(1+5k) on P(y=1|s); for the reward-ratio runs (nominal cut-off 1/(1+R)) it
  is 1/(1+R k).
* The standard DQN's +-1 reward with the Huber loss (threshold 1): at gamma = 0
  the minimiser of the expected loss for a two-point target is
  Q(s,1) = (2p-1)/max(p,1-p) = -Q(s,0), so p_hat = 1/2 + (2p-1)/(2 max(p,1-p))
  and p_hat = 1/6 at p = 1/4.  For gamma > 0 there is no closed form.
* Models trained on the natural class mix with a class weight of 5 (logistic
  regression, HGB, random forest, CS-MLP with uniform minibatches) and unweighted
  by Eq. (7) keep the nominal cut-off 1/6.

It also records the observed frame-rate range of each game after the first
second (the logger writes 0 fps at start-up), which the paper uses to state
where the frame-rate caps of the perturbation scenarios lie.

Sensitivity (sensitivity()): rule D1 and the ranking metrics do not use the
nominal cut-off, but they are not strictly invariant to the shift, because k
differs between the three inner models whose validation scores are pooled (D1)
and whose test scores are averaged, and between folds.  For E-DDQN, E-CB and the
CS-MLP, every inner model's validation and test scores are mapped to the scale of
P(y=1|s) with its own k (odds divided by k; a C1 linear continuation outside
(eps, 1-eps) keeps the map monotone for implied probabilities outside (0,1)),
D1 is re-derived exactly as in scripts/04_aggregate.py, and decisions, cost and
ROC-AUC are compared with the unchanged pipeline, which is first reproduced
exactly from the saved runs.  Requires results/runs/final (written by the
protocol run).

Usage: python scripts/14_score_scale.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn import CONFIGS, RESULTS  # noqa: E402
from eddqn.data import build_datasets, fold_frames, split_tables  # noqa: E402

PROTOCOL = json.loads((CONFIGS / "protocol.json").read_text())
OUT = RESULTS / "tables" / "score_scale"
FAMS = ["E-DDQN", "E-CB", "DQN", "CS-MLP", "LogReg", "HGB", "RF"]


def huber_pm1_phat(p):
    """Implied probability of the +-1 reward under the Huber loss (threshold 1) at gamma = 0."""
    p = np.asarray(p, float)
    return 0.5 + (2 * p - 1) / (2 * np.maximum(p, 1 - p))


def huber_cutoff():
    grid = np.linspace(1e-4, 1 - 1e-4, 200001)
    return float(grid[np.argmin(np.abs(huber_pm1_phat(grid) - 1 / 6))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    long, _, _ = build_datasets()
    tables = split_tables(long, PROTOCOL)
    sel = json.loads((RESULTS / "selected_configs.json").read_text())
    p_huber = huber_cutoff()
    rows = []
    for k in range(PROTOCOL["k_outer"]):
        pis = [float(fold_frames(long, tables, k, j)[0]["y"].mean()) for j in range(PROTOCOL["k_inner"])]
        for fam in FAMS:
            c = sel[fam][str(k)]["config"]
            rho, regime, ks = None, "natural class mix, class weight 5", None
            if fam in ("E-DDQN", "E-CB"):
                rho = float(c["rho"])
                regime = "class-balanced replay"
                ks = [((1 - rho) / rho) * ((1 - pi) / pi) for pi in pis]
            elif fam == "CS-MLP" and c.get("sampling") == "balanced":
                regime = "class-balanced minibatches (50/50), class weight 5"
                ks = [(1 - pi) / pi for pi in pis]
            elif fam == "DQN":
                regime = "+-1 reward, Huber loss, uniform replay"
            if ks is not None:
                cut = [1 / (1 + 5 * kk) for kk in ks]
            elif fam == "DQN":
                cut = [p_huber] * len(pis) if float(c["gamma"]) == 0.0 else [np.nan] * len(pis)
            else:
                cut = [1 / 6] * len(pis)
            rows.append({"family": fam, "outer": k, "regime": regime, "rho": rho,
                         "gamma": c.get("gamma"), "train_prevalence_mean": float(np.mean(pis)),
                         "train_prevalence_min": float(np.min(pis)), "train_prevalence_max": float(np.max(pis)),
                         "k_mean": float(np.mean(ks)) if ks is not None else np.nan,
                         "k_min": float(np.min(ks)) if ks is not None else np.nan,
                         "k_max": float(np.max(ks)) if ks is not None else np.nan,
                         "effective_cutoff_mean": float(np.mean(cut)) if not np.isnan(cut).all() else np.nan,
                         "effective_cutoff_min": float(np.min(cut)) if not np.isnan(cut).all() else np.nan,
                         "effective_cutoff_max": float(np.max(cut)) if not np.isnan(cut).all() else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "effective_cutoffs.csv", index=False)
    fps = {g: (int(d[d.step > 0].GameFps.min()), int(d[d.step > 0].GameFps.max())) for g, d in long.groupby("game")}
    pd.DataFrame([{"game": g, "fps_min_after_first_second": a, "fps_max": b} for g, (a, b) in fps.items()]).to_csv(
        OUT / "frame_rate_range.csv", index=False)
    summ = sensitivity(long, tables, df)
    write_tables_and_macros(df, p_huber, fps, summ)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _to_p_scale(p, k, eps=1e-3):
    """Map a score whose odds are k times those of P(y=1|s) to the P scale (odds / k),
    monotone on the whole real line (C1 linear continuation outside (eps, 1-eps))."""
    p = np.asarray(p, float)
    lk = np.log(k)
    f = lambda x: 1 / (1 + np.exp(-(np.log(x / (1 - x)) - lk)))
    fp = lambda x: f(x) * (1 - f(x)) / (x * (1 - x))
    out = np.empty_like(p)
    lo, hi = p < eps, p > 1 - eps
    mid = ~(lo | hi)
    out[mid] = f(p[mid])
    out[lo] = f(eps) + (p[lo] - eps) * fp(eps)
    out[hi] = f(1 - eps) + (p[hi] - 1 + eps) * fp(1 - eps)
    return out


def sensitivity(long, tables, df):
    """Effect on rule D1 and ranking metrics of rescaling every inner model to P(y=1|s)."""
    import pickle
    from sklearn.metrics import roc_auc_score
    from eddqn.metrics import threshold_min_cost
    runs = RESULTS / "runs" / "final"
    if not runs.exists():
        print("results/runs/final not found: sensitivity skipped")
        return None
    seeds = PROTOCOL["final_seeds"]
    yv = {(k, j): fold_frames(long, tables, k, j)[1]["y"].to_numpy() for k in range(5) for j in range(3)}
    yt = {k: fold_frames(long, tables, k)[1]["y"].to_numpy() for k in range(5)}
    kk = {}
    for k in range(5):
        for j in range(3):
            pi = float(fold_frames(long, tables, k, j)[0]["y"].mean())
            for fam in ("E-DDQN", "E-CB", "CS-MLP"):
                c = json.loads((RESULTS / "selected_configs.json").read_text())[fam][str(k)]["config"]
                if fam == "CS-MLP":
                    kk[(fam, k, j)] = (1 - pi) / pi if c.get("sampling") == "balanced" else 1.0
                else:
                    kk[(fam, k, j)] = ((1 - float(c["rho"])) / float(c["rho"])) * ((1 - pi) / pi)
    rows = []
    for fam in ("E-DDQN", "E-CB", "CS-MLP"):
        saved = pd.read_csv(RESULTS / "predictions" / f"{fam}.csv.gz", dtype={"seed": str})
        for s in seeds:
            Y, A0, A1, S0, S1, W0, W1 = [], [], [], [], [], [], []
            for k in range(5):
                parts = [pickle.load(open(runs / fam / f"o{k}_s{s}_i{j}.pkl", "rb")) for j in range(3)]
                y_oof = np.concatenate([yv[(k, j)] for j in range(3)])
                # unchanged pipeline (as scripts/04_aggregate.py: pooled OOF threshold, mean test score)
                oof0 = pd.Series(np.concatenate([p["val_score"] for p in parts])).to_numpy()
                t0 = np.mean([p["test_score"] for p in parts], axis=0)
                a0 = (t0 >= threshold_min_cost(oof0, y_oof)).astype(int)
                ref = saved[(saved.seed == str(s)) & (saved.outer == k)].pred_D1_min_cost.to_numpy()
                assert np.array_equal(a0, ref), f"pipeline not reproduced: {fam} seed {s} fold {k}"
                # every inner model rescaled to P(y=1|s) with its own k
                oof1 = np.concatenate([_to_p_scale(p["val_score"], kk[(fam, k, j)]) for j, p in enumerate(parts)])
                t1 = np.mean([_to_p_scale(p["test_score"], kk[(fam, k, j)]) for j, p in enumerate(parts)], axis=0)
                a1 = (t1 >= threshold_min_cost(oof1, y_oof)).astype(int)
                Y.append(yt[k]); A0.append(a0); A1.append(a1); S0.append(t0); S1.append(t1)
                W0.append(roc_auc_score(yt[k], t0)); W1.append(roc_auc_score(yt[k], t1))
            y = np.concatenate(Y); a0, a1 = np.concatenate(A0), np.concatenate(A1)
            cost = lambda a: (5 * ((a == 0) & (y == 1)).sum() + ((a == 1) & (y == 0)).sum()) / len(y)
            rows.append({"family": fam, "seed": s, "decisions_changed": float((a0 != a1).mean()),
                         "cost_pipeline": cost(a0), "cost_rescaled": cost(a1),
                         "pooled_auc_pipeline": roc_auc_score(y, np.concatenate(S0)),
                         "pooled_auc_rescaled": roc_auc_score(y, np.concatenate(S1)),
                         "within_fold_auc_pipeline": float(np.mean(W0)), "within_fold_auc_rescaled": float(np.mean(W1))})
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "rescaling_sensitivity_per_seed.csv", index=False)
    summ = sens.groupby("family", sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    summ.to_csv(OUT / "rescaling_sensitivity.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summ


def write_tables_and_macros(df, p_huber, fps, summ=None):
    def f3(x):
        return "--" if pd.isna(x) else f"{x:.3f}"
    name = {"E-DDQN": "E-DDQN", "E-CB": "E-CB", "DQN": "DQN", "CS-MLP": "CS-MLP"}
    body = ["\\begin{tabular}{llccccc}", "\\toprule",
            "Model & Training class mix (selected) & Fold & $\\rho$ & $\\pi$ & $k$ & Cut-off on $P(y=1\\mid s)$: mean (range) \\\\", "\\midrule"]
    for fam in ("E-DDQN", "E-CB", "CS-MLP", "DQN"):
        for r in df[df.family == fam].itertuples():
            regime = r.regime if fam != "DQN" else f"$\\pm1$ reward, Huber loss ($\\gamma={r.gamma}$)"
            body.append(f"{name[fam]} & {regime} & {r.outer} & {f3(r.rho) if r.rho is not None and not pd.isna(r.rho) else '--'} & "
                        f"{r.train_prevalence_mean:.3f} & {f3(r.k_mean)} & {f3(r.effective_cutoff_mean)}"
                        + (f" ({r.effective_cutoff_min:.3f}--{r.effective_cutoff_max:.3f})" if not pd.isna(r.k_mean) else "") + " \\\\")
        body.append("\\midrule")
    body[-1] = "\\bottomrule"
    body.append("\\end{tabular}")
    (OUT / "tab_ss_cutoffs.tex").write_text("\n".join(body) + "\n")
    M = {}

    def put(k, v):
        assert k.isalpha(), k
        M["SS" + k] = v
    for fam, key in (("E-DDQN", "EDDQN"), ("E-CB", "ECB")):
        d = df[df.family == fam]
        put(key + "CutMin", f"{d.effective_cutoff_min.min():.3f}")
        put(key + "CutMax", f"{d.effective_cutoff_max.max():.3f}")
        put(key + "KMin", f"{d.k_min.min():.2f}")      # over all inner models of the selected configurations
        put(key + "KMax", f"{d.k_max.max():.2f}")
    d = df[(df.family == "CS-MLP") & df.k_mean.notna()]
    put("CSMLPBalancedFolds", str(len(d)))
    put("CSMLPCutMin", f"{d.effective_cutoff_min.min():.3f}")
    put("CSMLPCutMax", f"{d.effective_cutoff_max.max():.3f}")
    shifted = df[df.k_mean.notna()]
    put("ShiftCutMin", f"{shifted.effective_cutoff_min.min():.2f}")
    put("ShiftCutMax", f"{shifted.effective_cutoff_max.max():.2f}")
    put("DQNHuberCut", f"{p_huber:.2f}")
    put("DQNGammaZeroFolds", str(int((df[df.family == "DQN"].gamma.astype(float) == 0).sum())))
    for g, (a, b) in fps.items():
        put(f"Fps{g}Min", str(a))
        put(f"Fps{g}Max", str(b))
    put("PrevMin", f"{df.train_prevalence_min.min():.3f}")   # over all inner training splits
    put("PrevMax", f"{df.train_prevalence_max.max():.3f}")
    if summ is not None:
        put("SensMaxChanged", f"{100 * summ.decisions_changed.max():.1f}")
        put("SensMaxCostDelta", f"{(summ.cost_rescaled - summ.cost_pipeline).abs().max():.3f}")
        put("SensMaxAucDelta", f"{(summ.pooled_auc_rescaled - summ.pooled_auc_pipeline).abs().max():.3f}")
        put("SensMaxWfDelta", f"{(summ.within_fold_auc_rescaled - summ.within_fold_auc_pipeline).abs().max():.3f}")
        body = ["\\begin{tabular}{lccccccc}", "\\toprule",
                "Model & D1 decisions changed (\\%) & \\multicolumn{2}{c}{Cost$_{5:1}$} & \\multicolumn{2}{c}{Pooled ROC-AUC} & \\multicolumn{2}{c}{Within-fold ROC-AUC} \\\\",
                " & & pipeline & rescaled & pipeline & rescaled & pipeline & rescaled \\\\", "\\midrule"]
        for r in summ.itertuples():
            body.append(f"{r.family} & {100 * r.decisions_changed:.2f} & {r.cost_pipeline:.3f} & {r.cost_rescaled:.3f} & "
                        f"{r.pooled_auc_pipeline:.3f} & {r.pooled_auc_rescaled:.3f} & {r.within_fold_auc_pipeline:.3f} & {r.within_fold_auc_rescaled:.3f} \\\\")
        body += ["\\bottomrule", "\\end{tabular}"]
        (OUT / "tab_ss_sensitivity.tex").write_text("\n".join(body) + "\n")
    (OUT / "ss_macros.json").write_text(json.dumps(M, indent=1))
    (OUT / "ss_macros.tex").write_text("% generated by scripts/14_score_scale.py -- do not edit\n" +
                                       "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")


if __name__ == "__main__":
    main()
