"""Aggregate the frozen final runs into predictions, metrics, statistics and figures.

Every number in the revised manuscript is produced here from the saved
per-job scores (results/runs/final/**) and written to results/tables/*.csv,
results/predictions/*.csv.gz and results/figures/*.pdf|png.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn import CONFIGS, PROCESSED, RESULTS  # noqa: E402
from eddqn.data import build_datasets, fold_frames, split_tables  # noqa: E402
from eddqn.experiment import operating_thresholds  # noqa: E402
from eddqn.metrics import (all_metrics, binary_metrics, cluster_bootstrap, cost_of, mcnemar,  # noqa: E402
                           paired_cluster_bootstrap_diff)

PROTOCOL = json.loads((CONFIGS / "protocol.json").read_text())
RUNS = RESULTS / "runs" / "final"
TAB = RESULTS / "tables"
FIG = RESULTS / "figures"
PRED = RESULTS / "predictions"
for d in (TAB, FIG, PRED):
    d.mkdir(parents=True, exist_ok=True)

MAIN = ["E-DDQN", "E-CB", "DQN", "CS-MLP", "LogReg", "HGB", "RF"]
RULES = ["D1_min_cost", "D0_bayes_1_6", "recall>=0.95"]
SEEDS = PROTOCOL["final_seeds"]
KO, KI = PROTOCOL["k_outer"], PROTOCOL["k_inner"]
REPORT = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc", "g_mean",
          "fpr", "fnr", "FN", "FP", "TP", "TN", "expected_cost", "alarm_rate", "roc_auc", "pr_auc"]


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


class Frames:
    def __init__(self):
        self.long, _, _ = build_datasets()
        self.tables = split_tables(self.long, PROTOCOL)
        self.test = {k: fold_frames(self.long, self.tables, k)[1].reset_index(drop=True) for k in range(KO)}
        self.val = {(k, j): fold_frames(self.long, self.tables, k, j)[1].reset_index(drop=True)
                    for k in range(KO) for j in range(KI)}


def collect(tag, F: Frames):
    """Return per (outer, seed): oof frame, thresholds, test scores, attack scores."""
    out = {}
    d = RUNS / tag
    for k in range(KO):
        for s in SEEDS:
            parts = [load(d / f"o{k}_s{s}_i{j}.pkl") for j in range(KI)]
            oof = pd.concat([F.val[(k, j)][["session_id", "group_id", "step", "y"]].assign(score=p["val_score"], inner=j)
                             for j, p in zip(range(KI), parts)], ignore_index=True)
            test = np.mean([p["test_score"] for p in parts], axis=0)
            att = {}
            if parts[0].get("attack_scores"):
                for name in parts[0]["attack_scores"]:
                    att[name] = np.mean([p["attack_scores"][name] for p in parts], axis=0)
            diag = {"best_epoch": [p.get("best_epoch") for p in parts], "seconds": [p["seconds"] for p in parts],
                    "history": [p.get("history", []) for p in parts],
                    "val_q": [p.get("val_q") for p in parts], "test_q": [p.get("test_q") for p in parts]}
            out[(k, s)] = {"oof": oof, "test": test, "attacks": att, "diag": diag}
    return out


def predictions_for(tag, runs, F):
    """Per seed and for the seed-ensemble: thresholds from pooled inner OOF of
    the development groups of each outer fold, applied to the outer-test fold."""
    rows = []
    for k in range(KO):
        te = F.test[k][["session_id", "group_id", "game", "step", "TimeStamp", "y", "grade"]]
        oofs = []
        for s in SEEDS:
            r = runs[(k, s)]
            th = operating_thresholds(r["oof"]["score"].to_numpy(), r["oof"]["y"].to_numpy(), PROTOCOL)
            oofs.append(r["oof"]["score"].to_numpy())
            rows.append(te.assign(outer=k, seed=str(s), score=r["test"],
                                  **{f"thr_{n}": v for n, v in th.items()},
                                  **{f"pred_{n}": (r["test"] >= v).astype(int) for n, v in th.items()}))
        ens_oof = np.mean(oofs, axis=0)
        th = operating_thresholds(ens_oof, runs[(k, SEEDS[0])]["oof"]["y"].to_numpy(), PROTOCOL)
        ens = np.mean([runs[(k, s)]["test"] for s in SEEDS], axis=0)
        rows.append(te.assign(outer=k, seed="ens", score=ens, **{f"thr_{n}": v for n, v in th.items()},
                              **{f"pred_{n}": (ens >= v).astype(int) for n, v in th.items()}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(PRED / f"{tag.replace(' ', '_')}.csv.gz", index=False)
    return df


def metric_table(tag, pred):
    rows = []
    for rule in [r for r in pred.columns if r.startswith("pred_")]:
        rn = rule[5:]
        for seed, g in pred.groupby("seed"):
            m = all_metrics(g["score"].to_numpy(), g["y"].to_numpy(), 0.5)  # ranking part only
            m.update(binary_metrics(g[rule].to_numpy(), g["y"].to_numpy()))
            rows.append({"model": tag, "rule": rn, "seed": seed, **{c: m[c] for c in REPORT}})
    return pd.DataFrame(rows)


def summarise(mt):
    per_seed = mt[mt.seed != "ens"]
    agg = per_seed.groupby(["model", "rule"])[REPORT].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    ens = mt[mt.seed == "ens"].set_index(["model", "rule"])[REPORT].add_prefix("ens_")
    return agg.join(ens).reset_index()


def bootstrap_cis(tag, pred, rule="D1_min_cost", n_boot=2000):
    g = pred[pred.seed == "ens"].reset_index(drop=True)
    y, p, s, grp = g.y.to_numpy(), g[f"pred_{rule}"].to_numpy(), g.score.to_numpy(), g.group_id.to_numpy()
    from sklearn.metrics import average_precision_score, roc_auc_score
    def safe(fn):
        def f(idx):
            try:
                return fn(idx)
            except ValueError:
                return np.nan
        return f
    stats = {
        "recall": safe(lambda i: binary_metrics(p[i], y[i])["recall"]),
        "specificity": safe(lambda i: binary_metrics(p[i], y[i])["specificity"]),
        "balanced_accuracy": safe(lambda i: binary_metrics(p[i], y[i])["balanced_accuracy"]),
        "mcc": safe(lambda i: binary_metrics(p[i], y[i])["mcc"]),
        "f1": safe(lambda i: binary_metrics(p[i], y[i])["f1"]),
        "expected_cost": safe(lambda i: cost_of(p[i], y[i])),
        "roc_auc": safe(lambda i: roc_auc_score(y[i], s[i])),
        "pr_auc": safe(lambda i: average_precision_score(y[i], s[i])),
    }
    return {k: cluster_bootstrap(fn, grp, n_boot=n_boot, seed=11) for k, fn in stats.items()}


def trivial(F: Frames):
    rows = []
    allte = pd.concat([F.test[k].assign(outer=k) for k in range(KO)], ignore_index=True)
    for name, pred in (("Always alarm", np.ones(len(allte), int)), ("Never alarm", np.zeros(len(allte), int))):
        m = binary_metrics(pred, allte.y.to_numpy())
        rows.append({"model": name, "rule": "-", **{c: m.get(c, np.nan) for c in REPORT}})
    return pd.DataFrame(rows), allte


def matched_operating_points(preds: dict):
    """FPR/specificity of every model at validation-matched recall targets and
    recall at validation-matched specificity targets (thresholds from OOF)."""
    rows = []
    for tag, pr in preds.items():
        g = pr[pr.seed == "ens"]
        for col in [c for c in g.columns if c.startswith("pred_recall>=") or c.startswith("pred_spec>=")]:
            m = binary_metrics(g[col].to_numpy(), g.y.to_numpy())
            rows.append({"model": tag, "target (on validation OOF)": col[5:], "test_recall": m["recall"],
                         "test_specificity": m["specificity"], "test_fpr": m["fpr"], "FP": m["FP"], "FN": m["FN"],
                         "expected_cost": m["expected_cost"]})
    return pd.DataFrame(rows)


def oracle_matched(preds: dict, targets=(0.8, 0.9, 0.95, 0.99)):
    """Descriptive only (NOT a selection rule): FPR on the pooled test set at the
    test-set threshold that reaches each recall - i.e. points on the ROC curve."""
    rows = []
    for tag, pr in preds.items():
        g = pr[pr.seed == "ens"]
        fpr, tpr, _ = roc_curve(g.y, g.score)
        for t in targets:
            i = np.argmax(tpr >= t)
            rows.append({"model": tag, "recall_target": t, "fpr_at_target_on_roc": fpr[i]})
    return pd.DataFrame(rows)


def pairwise_tests(preds: dict, ref="E-DDQN", rule="D1_min_cost"):
    rows = []
    a = preds[ref]
    for tag, b in preds.items():
        if tag == ref:
            continue
        ga, gb = a[a.seed == "ens"].reset_index(drop=True), b[b.seed == "ens"].reset_index(drop=True)
        assert (ga[["session_id", "step"]].values == gb[["session_id", "step"]].values).all()
        y = ga.y.to_numpy()
        mc = mcnemar(ga[f"pred_{rule}"], gb[f"pred_{rule}"], y)
        cb = paired_cluster_bootstrap_diff(cost_of, ga[f"pred_{rule}"].to_numpy(), gb[f"pred_{rule}"].to_numpy(), y,
                                           ga.group_id.to_numpy(), n_boot=2000, seed=5)
        per_seed = []
        for s in SEEDS:
            sa, sb = a[a.seed == str(s)].reset_index(drop=True), b[b.seed == str(s)].reset_index(drop=True)
            per_seed.append(mcnemar(sa[f"pred_{rule}"], sb[f"pred_{rule}"], sa.y.to_numpy())["p_exact"])
        rows.append({"A": ref, "B": tag, "rule": rule, **mc,
                     "cost_A_minus_B": cb["diff"], "cost_diff_ci_low": cb["ci_low"], "cost_diff_ci_high": cb["ci_high"],
                     "cost_diff_p_cluster_bootstrap": cb["p_boot_two_sided"],
                     "mcnemar_p_exact_per_seed": ";".join(f"{p:.3g}" for p in per_seed)})
    return pd.DataFrame(rows)


def event_level(preds: dict, rule="D1_min_cost"):
    rows = []
    for tag, pr in preds.items():
        g = pr[pr.seed == "ens"]
        det, delays, pre_fa, neg_fa = [], [], [], []
        for sid, s in g.groupby("session_id"):
            s = s.sort_values("step")
            yv, pv, t = s.y.to_numpy(), s[f"pred_{rule}"].to_numpy(), s.TimeStamp.to_numpy()
            if yv.any():
                on = np.argmax(yv == 1)
                after = np.flatnonzero(pv[on:] == 1)
                det.append(len(after) > 0)
                if len(after):
                    delays.append(t[on + after[0]] - t[on])
                pre_fa.append(pv[:on].mean() if on > 0 else np.nan)
            else:
                neg_fa.append(pv.mean())
        rows.append({"model": tag, "sessions_with_event": len(det), "events_detected": int(np.sum(det)),
                     "median_detection_delay_s": float(np.median(delays)) if delays else np.nan,
                     "mean_pre_onset_alarm_fraction": float(np.nanmean(pre_fa)),
                     "sessions_without_event": len(neg_fa),
                     "mean_alarm_fraction_in_event_free_sessions": float(np.mean(neg_fa)) if neg_fa else np.nan})
    return pd.DataFrame(rows)


def curves(preds: dict, name="main"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for tag, pr in preds.items():
        g = pr[pr.seed == "ens"]
        fpr, tpr, _ = roc_curve(g.y, g.score)
        p, r, _ = precision_recall_curve(g.y, g.score)
        from sklearn.metrics import average_precision_score, roc_auc_score
        lw = 2.2 if tag == "E-DDQN" else 1.2
        ax[0].plot(fpr, tpr, lw=lw, label=f"{tag} (AUC {roc_auc_score(g.y, g.score):.3f})")
        ax[1].plot(r, p, lw=lw, label=f"{tag} (AP {average_precision_score(g.y, g.score):.3f})")
    prev = g.y.mean()
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[1].axhline(prev, color="k", ls=":", lw=0.8)
    ax[0].set(xlabel="False-positive rate", ylabel="Recall (true-positive rate)", title="ROC, pooled outer-test folds")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, pooled outer-test folds")
    for a in ax:
        a.legend(fontsize=7, loc="lower right" if a is ax[0] else "upper right")
        a.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"roc_pr_{name}.{ext}", dpi=200)
    plt.close(fig)


def confusion_figure(preds: dict, rule="D1_min_cost"):
    tags = list(preds)
    fig, axes = plt.subplots(1, len(tags), figsize=(2.3 * len(tags), 2.6))
    for ax, tag in zip(np.atleast_1d(axes), tags):
        g = preds[tag][preds[tag].seed == "ens"]
        m = binary_metrics(g[f"pred_{rule}"], g.y)
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["silent", "alarm"], fontsize=7)
        ax.set_yticks([0, 1], ["y=0", "y=1"], fontsize=7)
        ax.set_title(tag, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"confusion_{rule}.{ext}", dpi=200)
    plt.close(fig)


def attack_auc(runs_by_tag: dict, F: Frames):
    """Threshold-free robustness (descriptive): pooled ROC-AUC / PR-AUC of the
    seed-ensemble score on clean vs. perturbed test telemetry (whole-session
    variants), and the D0 (untuned 1/6) operating point under each variant."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    masks = load(PROCESSED / "attack_masks.pkl")
    var = [v for v in masks["variants"] if v["window"] == "whole"]
    rows = []
    for tag, runs in runs_by_tag.items():
        if not runs[(0, SEEDS[0])]["attacks"]:
            continue
        y = np.concatenate([F.test[k].y.to_numpy() for k in range(KO)])
        clean = np.concatenate([np.mean([runs[(k, s)]["test"] for s in SEEDS], axis=0) for k in range(KO)])
        base = {"model": tag, "variant": "clean", "kind": "clean", "value": np.nan,
                "roc_auc": roc_auc_score(y, clean), "pr_auc": average_precision_score(y, clean),
                **{f"D0_{a}": b for a, b in binary_metrics((clean >= 1 / 6).astype(int), y).items() if a in ("recall", "fpr", "expected_cost")}}
        rows.append(base)
        for v in var:
            sc = np.concatenate([np.mean([runs[(k, s)]["attacks"][v["name"]] for s in SEEDS], axis=0) for k in range(KO)])
            rows.append({"model": tag, "variant": v["name"], "kind": v["kind"], "value": v["value"],
                         "roc_auc": roc_auc_score(y, sc), "pr_auc": average_precision_score(y, sc),
                         **{f"D0_{a}": b for a, b in binary_metrics((sc >= 1 / 6).astype(int), y).items() if a in ("recall", "fpr", "expected_cost")}})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "attack_threshold_free.csv", index=False)
    return df


def optimism_gap(sel: dict):
    """Selected configuration's pooled inner-OOF cost (HPO seed) vs. outer-test
    cost of the same family (seed mean, D1) per outer fold."""
    pf = pd.read_csv(TAB / "metrics_per_outer_fold_D1.csv")
    rows = []
    for fam, d in sel.items():
        for k, v in d.items():
            t = pf[(pf.model == fam) & (pf.outer == int(k))]
            if t.empty:
                continue
            rows.append({"model": fam, "outer": int(k), "inner_oof_cost_at_selection": v["pooled_oof_min_cost"],
                         "outer_test_cost": float(t.expected_cost.iloc[0])})
    df = pd.DataFrame(rows)
    df["optimism_gap"] = df.outer_test_cost - df.inner_oof_cost_at_selection
    df.to_csv(TAB / "optimism_gap.csv", index=False)
    return df


def attack_tables(runs_by_tag: dict, preds: dict, F: Frames):
    masks = load(PROCESSED / "attack_masks.pkl")
    var = masks["variants"]
    rows, worst = [], []
    for tag, runs in runs_by_tag.items():
        if not runs[(0, SEEDS[0])]["attacks"]:
            continue
        pr = preds[tag]
        for k in range(KO):
            te = F.test[k]
            y = te.y.to_numpy()
            thr = pr[(pr.seed == "ens") & (pr.outer == k)]["thr_D1_min_cost"].iloc[0]
            clean = np.mean([runs[(k, s)]["test"] for s in SEEDS], axis=0)
            for v in var:
                z = masks["z"][(k, v["name"])]
                sc = np.mean([runs[(k, s)]["attacks"][v["name"]] for s in SEEDS], axis=0)
                rows.append({"model": tag, "outer": k, "variant": v["name"], "kind": v["kind"], "value": v["value"],
                             "window": v["window"], "role": v["role"],
                             "n_attacked": int(z.sum()), "n_attacked_pos": int((z & (y == 1)).sum()),
                             "alarms_clean_pos": int(((clean >= thr) & z & (y == 1)).sum()),
                             "alarms_attacked_pos": int(((sc >= thr) & z & (y == 1)).sum()),
                             "alarms_clean_neg": int(((clean >= thr) & z & (y == 0)).sum()),
                             "alarms_attacked_neg": int(((sc >= thr) & z & (y == 0)).sum()),
                             "n_attacked_neg": int((z & (y == 0)).sum()),
                             "mean_abs_score_change": float(np.abs(sc - clean)[z].mean())})
            # adaptive worst case per session (whole-session candidates incl. evasion-only)
            cands = [v for v in var if v["window"] == "whole"]
            for sid in te.session_id.unique():
                idx = (te.session_id == sid).to_numpy()
                pos, neg = idx & (y == 1), idx & (y == 0)
                base_pos = int((clean[pos] >= thr).sum())
                base_neg = int((clean[neg] >= thr).sum())
                best_ev, best_fa = base_pos, base_neg
                for v in cands:
                    sc = np.mean([runs[(k, s)]["attacks"][v["name"]] for s in SEEDS], axis=0)
                    best_ev = min(best_ev, int((sc[pos] >= thr).sum()))
                    best_fa = max(best_fa, int((sc[neg] >= thr).sum()))
                worst.append({"model": tag, "outer": k, "session_id": sid, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                              "alarms_pos_clean": base_pos, "alarms_pos_worst_evasion": best_ev,
                              "alarms_neg_clean": base_neg, "alarms_neg_worst_false_alarm": best_fa})
    a = pd.DataFrame(rows)
    a.to_csv(TAB / "attack_variants_by_fold.csv", index=False)
    agg = a.groupby(["model", "kind", "value", "window", "role"]).sum(numeric_only=True).reset_index()
    agg["recall_clean"] = agg.alarms_clean_pos / agg.n_attacked_pos.clip(lower=1)
    agg["recall_attacked"] = agg.alarms_attacked_pos / agg.n_attacked_pos.clip(lower=1)
    agg["fpr_clean"] = agg.alarms_clean_neg / agg.n_attacked_neg.clip(lower=1)
    agg["fpr_attacked"] = agg.alarms_attacked_neg / agg.n_attacked_neg.clip(lower=1)
    agg["cost_clean"] = (5 * (agg.n_attacked_pos - agg.alarms_clean_pos) + agg.alarms_clean_neg) / agg.n_attacked
    agg["cost_attacked"] = (5 * (agg.n_attacked_pos - agg.alarms_attacked_pos) + agg.alarms_attacked_neg) / agg.n_attacked
    agg.drop(columns=["outer"]).to_csv(TAB / "attack_summary.csv", index=False)
    w = pd.DataFrame(worst)
    ws = w.groupby("model").sum(numeric_only=True)
    ws = pd.DataFrame({"recall_clean": ws.alarms_pos_clean / ws.n_pos,
                       "recall_worst_case_evasion": ws.alarms_pos_worst_evasion / ws.n_pos,
                       "fpr_clean": ws.alarms_neg_clean / ws.n_neg,
                       "fpr_worst_case_false_alarm": ws.alarms_neg_worst_false_alarm / ws.n_neg}).reset_index()
    ws.to_csv(TAB / "attack_worst_case.csv", index=False)
    return agg, ws


def diagnostics(runs_by_tag: dict, F: Frames):
    rows = []
    for tag, runs in runs_by_tag.items():
        for (k, s), r in runs.items():
            d = r["diag"]
            for j in range(KI):
                rec = {"model": tag, "outer": k, "seed": s, "inner": j, "best_epoch": d["best_epoch"][j],
                       "seconds": d["seconds"][j]}
                q = d["val_q"][j]
                if q is not None:
                    val = F.val[(k, j)]
                    rec["q_abs_max"] = float(np.abs(q).max())
                    rec["val_alarm_rate_argmax"] = float((q[:, 1] >= q[:, 0]).mean())
                    rec["val_pos_rate"] = float(val.y.mean())
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "training_diagnostics.csv", index=False)
    return df


def overestimation(runs_by_tag: dict, F: Frames, sel: dict):
    """Mean Q(s, greedy a) minus the realised discounted return of the greedy
    policy on validation sessions (rewards are known offline for any action)."""
    from eddqn.agents import reward_table
    rows = []
    for tag in ("E-DDQN", "DQN", "E-CB"):
        if tag not in runs_by_tag:
            continue
        for (k, s), r in runs_by_tag[tag].items():
            cfg = sel[tag][str(k)]["config"]
            gamma = cfg.get("gamma", 0.0)
            rt = reward_table(cfg["reward"])
            for j in range(KI):
                q = r["diag"]["val_q"][j]
                val = F.val[(k, j)]
                a = (q[:, 1] >= q[:, 0]).astype(int)
                rew = np.where(a == 1, np.where(val.y == 1, rt["tp"], rt["fp"]), np.where(val.y == 1, rt["fn"], rt["tn"]))
                G = np.zeros(len(val))
                for sid, idx in val.groupby("session_id").indices.items():
                    idx = np.sort(idx)
                    acc = 0.0
                    for i in idx[::-1]:
                        acc = rew[i] + gamma * acc
                        G[i] = acc
                qa = q[np.arange(len(q)), a]
                rows.append({"model": tag, "outer": k, "seed": s, "inner": j, "gamma": gamma,
                             "mean_Q_greedy": float(qa.mean()), "mean_MC_return": float(G.mean()),
                             "overestimation": float((qa - G).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "overestimation.csv", index=False)
    return df


def ratio_argmax():
    """Policy induced by each reward ratio R: greedy arg-max Q, i.e. implied
    P(y=1|s) >= 1/(1+R) (no threshold tuning). Mean over seeds."""
    rows = []
    for f in sorted(PRED.glob("RATIO_*.csv.gz")):
        r = float(f.name.split("_")[1].split(".")[0])
        df = pd.read_csv(f)
        for seed, g in df[df.seed != "ens"].groupby("seed"):
            m = binary_metrics((g.score >= 1 / (1 + r)).astype(int), g.y)
            rows.append({"ratio": r, "seed": seed, **{k: m[k] for k in ("recall", "specificity", "fpr", "expected_cost", "alarm_rate")}})
    if rows:
        d = pd.DataFrame(rows).drop(columns="seed").groupby("ratio").agg(["mean", "std"])
        d.columns = [f"{a}_{b}" for a, b in d.columns]
        d.reset_index().to_csv(TAB / "ratio_argmax_policy.csv", index=False)


if __name__ == "__main__":
    F = Frames()
    sel = json.loads((RESULTS / "selected_configs.json").read_text())
    tags = [t for t in MAIN if (RUNS / t).exists()]
    runs = {t: collect(t, F) for t in tags}
    preds = {t: predictions_for(t, runs[t], F) for t in tags}
    mt = pd.concat([metric_table(t, preds[t]) for t in tags])
    mt.to_csv(TAB / "metrics_per_seed.csv", index=False)
    summ = summarise(mt)
    triv, _ = trivial(F)
    summ.to_csv(TAB / "metrics_summary.csv", index=False)
    triv.to_csv(TAB / "trivial_baselines.csv", index=False)
    cis = {t: bootstrap_cis(t, preds[t]) for t in tags}
    (TAB / "bootstrap_cis_D1.json").write_text(json.dumps(cis, indent=1))
    matched_operating_points(preds).to_csv(TAB / "matched_operating_points.csv", index=False)
    oracle_matched(preds).to_csv(TAB / "roc_points_at_recall_targets_descriptive.csv", index=False)
    if "E-DDQN" in preds:
        for rule in ("D1_min_cost", "D0_bayes_1_6", "recall>=0.95"):
            pairwise_tests(preds, rule=rule).to_csv(TAB / f"paired_tests_{rule.replace('>=', 'ge')}.csv", index=False)
    event_level(preds).to_csv(TAB / "event_level_D1.csv", index=False)
    curves(preds)
    confusion_figure(preds)
    per_fold = []
    for t in tags:
        g = preds[t][preds[t].seed == "ens"]
        for k, gk in g.groupby("outer"):
            m = all_metrics(gk.score.to_numpy(), gk.y.to_numpy(), gk.thr_D1_min_cost.iloc[0])
            per_fold.append({"model": t, "outer": k, **{c: m[c] for c in REPORT + ["threshold"]}})
    pd.DataFrame(per_fold).to_csv(TAB / "metrics_per_outer_fold_D1.csv", index=False)
    diagnostics(runs, F)
    overestimation(runs, F, sel)
    if (PROCESSED / "attack_masks.pkl").exists():
        attack_tables(runs, preds, F)
        attack_auc(runs, F)
    optimism_gap(sel)
    # ablations and reward-ratio sensitivity
    complete = KO * len(SEEDS) * KI
    extra = [p.name for p in RUNS.iterdir() if p.name.startswith(("ABL ", "RATIO ")) and len(list(p.glob("*.pkl"))) == complete] \
        if RUNS.exists() else []
    if extra:
        er = {t: collect(t, F) for t in extra}
        ep = {t: predictions_for(t, er[t], F) for t in extra}
        emt = pd.concat([metric_table(t, ep[t]) for t in extra])
        emt.to_csv(TAB / "ablation_ratio_metrics_per_seed.csv", index=False)
        summarise(emt).to_csv(TAB / "ablation_ratio_summary.csv", index=False)
    ratio_argmax()
    print(summ[summ.rule == "D1_min_cost"][["model", "recall_mean", "specificity_mean", "expected_cost_mean",
                                             "roc_auc_mean", "pr_auc_mean", "mcc_mean"]].round(3).to_string())
