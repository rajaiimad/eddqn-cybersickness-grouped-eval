"""Search-draw sensitivity of the E-DDQN vs. standard-DQN comparison
(configs/posthoc.json, key "search_draw_sensitivity").

Specified after all other results were known and committed before any of
these runs.  Each variant repeats the frozen protocol of configs/protocol.json
(splits, inner folds, search training seed, budget, selection, checkpoint,
cross-fitting, final seeds and threshold rules) with a different random draw
of configurations or with the feature set fixed to "exposure".  The primary
results are never replaced; every variant is reported.

Usage:
  python scripts/13_search_draw_sensitivity.py run [--workers N]
  python scripts/13_search_draw_sensitivity.py analyze
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from eddqn import CONFIGS, RESULTS  # noqa: E402
from eddqn import experiment as EXP  # noqa: E402
from eddqn.metrics import cost_of, min_cost_value, paired_cluster_bootstrap_diff, ranking_metrics  # noqa: E402


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # pool workers pickle functions by module name
    spec.loader.exec_module(mod)
    return mod


RP = _module("run_protocol", "03_run_protocol.py")
AG = _module("aggregate", "04_aggregate.py")
PRIMARY_CONFIGS_FOR = RP.configs_for
PROTOCOL = RP.PROTOCOL
SPEC = json.loads((CONFIGS / "posthoc.json").read_text())["search_draw_sensitivity"]
VARIANTS = {v["name"]: v for v in SPEC["variants"]}
PAIRS = SPEC["pairs"]
KO, KI, SEEDS = AG.KO, AG.KI, AG.SEEDS
OUT = RESULTS / "tables" / "search_draw"
SEL_PATH = RESULTS / "selected_configs_search_draw.json"
HPO_PATH = RESULTS / "hpo_results_search_draw.csv"
NB = 2000
EXP.POSTHOC_BASE.update({n: v["base"] for n, v in VARIANTS.items()})   # variant name -> model family


def configs_for(name):
    """Primary families keep their primary draw; variants draw from the same
    search space with their own seed (and, if given, a fixed feature set)."""
    if name not in VARIANTS:
        return PRIMARY_CONFIGS_FOR(name)
    v = VARIANTS[name]
    space, fixed = RP.family_space(v["base"])
    space = dict(space)
    if v.get("feature_set"):
        space["feature_set"] = [v["feature_set"]]
    return EXP.sample_configs(space, fixed, v["budget"], v["config_seed"])


# ------------------------------------------------------------------ run
def run(names, workers):
    from eddqn.data import build_datasets, fold_frames, split_tables
    long, _, _ = build_datasets()
    tables = split_tables(long, PROTOCOL)
    saved = pd.read_csv(RP.PROCESSED / "outer_folds.csv")
    assert saved[["group_id", "fold"]].sort_values("group_id").values.tolist() == \
        tables["outer"][["group_id", "fold"]].sort_values("group_id").values.tolist(), "split drift"
    RP.configs_for = configs_for
    RP.stage_hpo(long, tables, names, workers)
    # selection: the primary rule (pooled inner-OOF minimum 5:1 cost, ties PR-AUC)
    sel = json.loads(SEL_PATH.read_text()) if SEL_PATH.exists() else {}
    rows = []
    for name in names:
        sel[name] = {}
        for k in range(KO):
            y = np.concatenate([fold_frames(long, tables, k, j)[1]["y"].to_numpy() for j in range(KI)])
            best = None
            for ci, cfg in enumerate(configs_for(name)):
                parts = [RP.load(RP.RUNS / "hpo" / name / f"o{k}_c{ci:02d}_i{j}.pkl") for j in range(KI)]
                s = np.concatenate([p["val_score"] for p in parts])
                crit, rk = min_cost_value(s, y), ranking_metrics(s, y)
                rows.append({"family": name, "outer": k, "config_id": ci, "pooled_oof_min_cost": crit,
                             "pooled_oof_pr_auc": rk["pr_auc"], "pooled_oof_roc_auc": rk["roc_auc"],
                             "mean_seconds": float(np.mean([p["seconds"] for p in parts])),
                             **{f"cfg_{a}": (str(b) if isinstance(b, (list, tuple)) else b) for a, b in cfg.items()}})
                key = (crit, -rk["pr_auc"])
                if best is None or key < best[0]:
                    best = (key, ci, cfg)
            sel[name][str(k)] = {"config_id": best[1], "config": best[2],
                                 "pooled_oof_min_cost": best[0][0], "pooled_oof_pr_auc": -best[0][1]}
    SEL_PATH.write_text(json.dumps(sel, indent=2, default=str))
    hpo = pd.DataFrame(rows)
    if HPO_PATH.exists():
        prev = pd.read_csv(HPO_PATH)
        hpo = pd.concat([prev[~prev.family.isin(names)], hpo])
    hpo.to_csv(HPO_PATH, index=False)
    RP._final_cache(long, tables, with_attacks=False)
    jobs = []
    for name in names:
        jobs += RP.final_jobs(name, sel[name], score_attacks=False)
    RP.run_jobs(jobs, workers)


# -------------------------------------------------------------- helpers
def signflip_p(d, grp):
    """Exact two-sided sign-flip test of sum(d)/N with groups as units."""
    groups = np.unique(grp)
    D = np.array([d[grp == g].sum() for g in groups], float)
    signs = np.array(list(itertools.product((1, -1), repeat=len(groups) - 1)), dtype=np.int8)
    stat = D[0] + signs @ D[1:]
    return float((np.abs(stat) >= abs(D.sum()) - 1e-9).mean())


def paired_auc_diff(sa, sb, y, groups, n_boot, seed):
    from sklearn.metrics import roc_auc_score
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, sa) - roc_auc_score(y, sb)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
    d = np.asarray(d)
    return {"diff": float(obs), "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def ens(pred):
    return pred[pred.seed == "ens"].sort_values(["outer", "session_id", "step"]).reset_index(drop=True)


def cvec(pred, y):
    return 5 * ((pred == 0) & (y == 1)) + ((pred == 1) & (y == 0))


# -------------------------------------------------------------- analyze
KEY = {"E-DDQN": "EDDQNDrawOne", "E-DDQN-draw2": "EDDQNDrawTwo", "E-DDQN-draw3": "EDDQNDrawThree",
       "E-DDQN-exposure": "EDDQNExposure", "DQN": "DQNDrawOne", "DQN-draw2": "DQNDrawTwo",
       "DQN-draw3": "DQNDrawThree", "DQN-exposure": "DQNExposure"}
PAIRKEY = {"draw 1 (primary)": "One", "draw 2": "Two", "draw 3": "Three", "exposure only": "Exposure"}


def analyze():
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    F = AG.Frames()
    psel = json.loads((RESULTS / "selected_configs.json").read_text())
    vsel = json.loads(SEL_PATH.read_text())
    names = [n for pair in PAIRS for n in pair[:2]]
    preds = {}
    for n in names:
        if n in VARIANTS:
            preds[n] = AG.predictions_for(n, AG.collect(n, F), F)
        else:                                       # primary families: predictions are read, never rewritten
            preds[n] = pd.read_csv(AG.PRED / f"{n}.csv.gz", dtype={"seed": str})
    rows = []
    for n in names:
        p = preds[n]
        seeds = p[p.seed != "ens"]
        cost = [cost_of(g.pred_D1_min_cost.to_numpy(), g.y.to_numpy()) for _, g in seeds.groupby("seed")]
        auc = [roc_auc_score(g.y, g.score) for _, g in seeds.groupby("seed")]
        wf = [np.mean([roc_auc_score(h.y, h.score) for _, h in g.groupby("outer")]) for _, g in seeds.groupby("seed")]
        e = ens(p)
        sel = (vsel if n in VARIANTS else psel)[n]
        draw = configs_for(n)
        rows.append({"variant": n, "family": VARIANTS[n]["base"] if n in VARIANTS else n,
                     "cost_seed_mean": np.mean(cost), "cost_seed_sd": np.std(cost, ddof=1),
                     "roc_auc_seed_mean": np.mean(auc), "within_fold_roc_auc": np.mean(wf),
                     "ens_cost": cost_of(e.pred_D1_min_cost.to_numpy(), e.y.to_numpy()),
                     "exposure_only_candidates": sum(c["feature_set"] == "exposure" for c in draw),
                     "questionnaire_candidates": sum("profile" in c["feature_set"] for c in draw),
                     "folds_selecting_questionnaire": sum("profile" in sel[str(k)]["config"]["feature_set"] for k in range(KO)),
                     "selected_feature_sets": ";".join(sel[str(k)]["config"]["feature_set"] for k in range(KO)),
                     "selected_gamma": ";".join(str(sel[str(k)]["config"].get("gamma")) for k in range(KO))})
    vs = pd.DataFrame(rows)
    vs.to_csv(OUT / "variants_summary.csv", index=False)
    prs = []
    for a, b, label in PAIRS:
        ea, eb = ens(preds[a]), ens(preds[b])
        assert (ea[["session_id", "step"]].values == eb[["session_id", "step"]].values).all()
        y, grp = ea.y.to_numpy(), ea.group_id.to_numpy()
        pa, pb = ea.pred_D1_min_cost.to_numpy(), eb.pred_D1_min_cost.to_numpy()
        bc = paired_cluster_bootstrap_diff(cost_of, pa, pb, y, grp, n_boot=NB, seed=5)
        au = paired_auc_diff(ea.score.to_numpy(), eb.score.to_numpy(), y, grp, NB, seed=6)
        prs.append({"pair": label, "A": a, "B": b, "cost_A_minus_B": bc["diff"], "ci_low": bc["ci_low"],
                    "ci_high": bc["ci_high"], "p_boot": bc["p_boot_two_sided"],
                    "p_signflip": signflip_p(cvec(pa, y) - cvec(pb, y), grp),
                    "auc_A_minus_B": au["diff"], "auc_ci_low": au["ci_low"], "auc_ci_high": au["ci_high"], "auc_p": au["p"]})
    pr = pd.DataFrame(prs)
    pr.to_csv(OUT / "pairs.csv", index=False)
    # pre-specified decision rules (configs/posthoc.json)
    v = vs.set_index("variant")
    ed = v.loc[["E-DDQN", "E-DDQN-draw2", "E-DDQN-draw3"]]
    r1 = bool(((pr.cost_A_minus_B < 0) & (pr.ci_high < 0)).any())
    cost_span = float(ed.cost_seed_mean.max() - ed.cost_seed_mean.min())
    wf_span = float(ed.within_fold_roc_auc.max() - ed.within_fold_roc_auc.min())
    r2 = cost_span > 0.05 or wf_span > 0.05
    gap = float(v.loc["DQN-exposure", "within_fold_roc_auc"] - v.loc["E-DDQN-exposure", "within_fold_roc_auc"])
    r3 = "supported" if abs(gap) <= 0.03 else ("insufficient" if gap > 0.05 else "inconclusive")   # spec: "within 0.03 of DQN's"
    dec = {"R1_headline_challenged": r1, "R2_draw_dependent": r2, "R2_cost_span": cost_span,
           "R2_within_fold_auc_span": wf_span, "R3_exposure_gap_dqn_minus_eddqn": gap, "R3_outcome": r3}
    (OUT / "decisions.json").write_text(json.dumps(dec, indent=1))
    write_tables_and_macros(vs, pr, dec)
    print(json.dumps(dec, indent=1))


def write_tables_and_macros(vs, pr, dec):
    def f3(x):
        return f"{x:+.3f}".replace("-", "$-$")

    def pv(x):
        return "$<$0.001" if x < 0.001 else f"{x:.3f}"
    label = {"E-DDQN": "E-DDQN, draw 1 (primary)", "E-DDQN-draw2": "E-DDQN, draw 2", "E-DDQN-draw3": "E-DDQN, draw 3",
             "E-DDQN-exposure": "E-DDQN, exposure only", "DQN": "DQN, draw 1 (primary)", "DQN-draw2": "DQN, draw 2",
             "DQN-draw3": "DQN, draw 3", "DQN-exposure": "DQN, exposure only"}
    body = ["\\begin{tabular}{lccccccc}", "\\toprule",
            "Variant & E-only cand. & Q folds & Cost$_{5:1}$ (seed mean) & Ens.\\ cost & ROC-AUC & Within-fold AUC & Selected $\\gamma$ \\\\",
            "\\midrule"]
    for r in vs.itertuples():
        body.append(f"{label[r.variant]} & {r.exposure_only_candidates} & {r.folds_selecting_questionnaire}/5 & "
                    f"{r.cost_seed_mean:.3f}$\\pm${r.cost_seed_sd:.3f} & {r.ens_cost:.3f} & {r.roc_auc_seed_mean:.3f} & "
                    f"{r.within_fold_roc_auc:.3f} & {r.selected_gamma.replace(';', ', ')} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_sd_variants.tex").write_text("\n".join(body) + "\n")
    body = ["\\begin{tabular}{lcccccccc}", "\\toprule",
            "Draw & $\\Delta$Cost (E-DDQN$-$DQN) & 95\\,\\% CI & $p_{\\text{boot}}$ & $p_{\\text{sign-flip}}$ & $\\Delta$ROC-AUC & 95\\,\\% CI & $p_{\\text{boot}}$ \\\\",
            "\\midrule"]
    for r in pr.itertuples():
        body.append(f"{r.pair} & {f3(r.cost_A_minus_B)} & [{f3(r.ci_low)}, {f3(r.ci_high)}] & {pv(r.p_boot)} & {pv(r.p_signflip)} & "
                    f"{f3(r.auc_A_minus_B)} & [{f3(r.auc_ci_low)}, {f3(r.auc_ci_high)}] & {pv(r.auc_p)} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_sd_pairs.tex").write_text("\n".join(body) + "\n")
    M = {}

    def put(k, val):
        assert k.isalpha(), k
        M["SD" + k] = val
    for r in vs.itertuples():
        k = KEY[r.variant]
        put(k + "Cost", f"{r.cost_seed_mean:.3f}")
        put(k + "EnsCost", f"{r.ens_cost:.3f}")
        put(k + "Auc", f"{r.roc_auc_seed_mean:.3f}")
        put(k + "WfAuc", f"{r.within_fold_roc_auc:.3f}")
        put(k + "ExpCand", str(r.exposure_only_candidates))
        put(k + "QFolds", str(r.folds_selecting_questionnaire))
    for r in pr.itertuples():
        k = "Pair" + PAIRKEY[r.pair]
        put(k + "Diff", f3(r.cost_A_minus_B))
        put(k + "Low", f3(r.ci_low))
        put(k + "High", f3(r.ci_high))
        put(k + "P", pv(r.p_boot))
        put(k + "SfP", pv(r.p_signflip))
        put(k + "AucDiff", f3(r.auc_A_minus_B))
        put(k + "AucLow", f3(r.auc_ci_low))
        put(k + "AucHigh", f3(r.auc_ci_high))
        put(k + "AucP", pv(r.auc_p))
    v = vs.set_index("variant")
    ed = v.loc[["E-DDQN", "E-DDQN-draw2", "E-DDQN-draw3"]]
    dq = v.loc[["DQN", "DQN-draw2", "DQN-draw3"]]
    put("EDDQNCostMin", f"{ed.cost_seed_mean.min():.3f}")
    put("EDDQNCostMax", f"{ed.cost_seed_mean.max():.3f}")
    put("EDDQNWfMin", f"{ed.within_fold_roc_auc.min():.3f}")
    put("EDDQNWfMax", f"{ed.within_fold_roc_auc.max():.3f}")
    put("DQNCostMin", f"{dq.cost_seed_mean.min():.3f}")
    put("DQNCostMax", f"{dq.cost_seed_mean.max():.3f}")
    put("DQNWfMin", f"{dq.within_fold_roc_auc.min():.3f}")
    put("DQNWfMax", f"{dq.within_fold_roc_auc.max():.3f}")
    put("EDDQNDrawTwoStrongGammaFolds", str(v.loc["E-DDQN-draw2", "selected_gamma"].split(";").count("0.99")))
    put("CostDiffMin", f3(pr.cost_A_minus_B.min()))
    put("CostDiffMax", f3(pr.cost_A_minus_B.max()))
    put("CostPMin", pv(pr.p_boot.min()))
    put("CostSpan", f"{dec['R2_cost_span']:.3f}")
    put("WfSpan", f"{dec['R2_within_fold_auc_span']:.3f}")
    put("ExposureGap", f"{dec['R3_exposure_gap_dqn_minus_eddqn']:.3f}")
    put("ROne", "yes" if dec["R1_headline_challenged"] else "no")
    put("RTwo", "yes" if dec["R2_draw_dependent"] else "no")
    put("RThree", dec["R3_outcome"])
    (OUT / "sd_macros.json").write_text(json.dumps(M, indent=1))
    (OUT / "sd_macros.tex").write_text("% generated by scripts/13_search_draw_sensitivity.py -- do not edit\n" +
                                       "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyze"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="run only these variants (default: all)")
    a = ap.parse_args()
    if a.stage == "run":
        run(a.only or list(VARIANTS), a.workers)
    else:
        analyze()
