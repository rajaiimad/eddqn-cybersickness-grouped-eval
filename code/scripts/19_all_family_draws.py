"""All-family search-draw experiment and saved-prediction analyses
(configs/posthoc.json, key "all_family_search_draws").

Specified after all other results were known and committed before any of
these runs.  Draws 2 and 3 of E-CB, CS-MLP, LogReg, HGB and RF repeat the
frozen protocol with a different random draw of configurations, exactly as
scripts/13_search_draw_sensitivity.py does for E-DDQN and the standard DQN
(whose draws 2 and 3 are reused).  The primary results are never replaced.

Usage:
  python code/scripts/19_all_family_draws.py run [--workers N] [--only NAME ...]
  python code/scripts/19_all_family_draws.py analyze
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from eddqn import CONFIGS, RESULTS  # noqa: E402
from eddqn import experiment as EXP  # noqa: E402
from eddqn.metrics import cost_of, paired_cluster_bootstrap_diff  # noqa: E402


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # pool workers pickle functions by module name
    spec.loader.exec_module(mod)
    return mod


SD = _module("search_draw", "13_search_draw_sensitivity.py")
AG, RP = SD.AG, SD.RP
SPEC = json.loads((CONFIGS / "posthoc.json").read_text())["all_family_search_draws"]
NEW = {v["name"]: {**v, "budget": RP.PROTOCOL["hpo_budget"][v["base"]]} for v in SPEC["variants"]}
DRAWS = SPEC["draws"]
FAMILIES = list(DRAWS)                       # E-DDQN first
OTHERS = FAMILIES[1:]
OUT = RESULTS / "tables" / "all_draws"
NB = 2000
Z = 1.960 + 0.842                            # two-sided alpha 0.05, power 0.80

# the search-draw machinery of script 13, extended with the new variants and its own output files
SD.VARIANTS.update(NEW)
SD.SEL_PATH = RESULTS / "selected_configs_all_draws.json"
SD.HPO_PATH = RESULTS / "hpo_results_all_draws.csv"
EXP.POSTHOC_BASE.update({n: v["base"] for n, v in NEW.items()})
SEL_13 = RESULTS / "selected_configs_search_draw.json"


# -------------------------------------------------------------- helpers
def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty_like(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def load_pred(name, F):
    path = AG.PRED / f"{name}.csv.gz"
    if name in NEW and not path.exists():
        AG.predictions_for(name, AG.collect(name, F), F)       # writes results/predictions/<name>.csv.gz
    return pd.read_csv(path, dtype={"seed": str}).sort_values(["outer", "session_id", "step"]).reset_index(drop=True)


def seed_cost_vec(p):
    """Per-sample 5:1 cost averaged over the five seeds' own D1 decisions."""
    s = p[p.seed != "ens"]
    v = [SD.cvec(g.pred_D1_min_cost.to_numpy(), g.y.to_numpy()) for _, g in s.groupby("seed")]
    return np.mean(v, axis=0)


def boot_test(ca, cb, grp, seed=5):
    """Paired group bootstrap and exact sign-flip test of mean(ca - cb)."""
    uniq = np.unique(grp)
    idx_of = {g: np.flatnonzero(grp == g) for g in uniq}
    rng = np.random.default_rng(seed)
    d = ca - cb
    reps = np.array([d[np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])].mean()
                     for _ in range(NB)])
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(reps, 2.5)), "ci_high": float(np.percentile(reps, 97.5)),
            "p_boot": float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean()))), "boot_sd": float(reps.std(ddof=1)),
            "p_signflip": SD.signflip_p(d, grp)}


def auc_boot_sd(sa, sb, y, grp, seed=6):
    from sklearn.metrics import roc_auc_score
    uniq = np.unique(grp)
    idx_of = {g: np.flatnonzero(grp == g) for g in uniq}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(NB):
        idx = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if y[idx].min() != y[idx].max():
            reps.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
    return float(np.std(reps, ddof=1))


# ------------------------------------------------------------------ analyze
def analyze():
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    F = AG.Frames()
    psel = json.loads((RESULTS / "selected_configs.json").read_text())
    sel = {**json.loads(SEL_13.read_text()), **json.loads(SD.SEL_PATH.read_text()), **psel}
    names = [n for f in FAMILIES for n in DRAWS[f]]
    P = {n: load_pred(n, F) for n in names}
    ref = P["E-DDQN"]
    for n in names:
        assert (P[n][["session_id", "step"]].values == ref[["session_id", "step"]].values).all(), n
    E = {n: SD.ens(P[n]) for n in names}
    y, grp = E["E-DDQN"].y.to_numpy(), E["E-DDQN"].group_id.to_numpy()
    ens_cost = {n: SD.cvec(E[n].pred_D1_min_cost.to_numpy(), y) for n in names}

    # A. per variant
    rows = []
    for f in FAMILIES:
        for d, n in enumerate(DRAWS[f], start=1):
            s = P[n][P[n].seed != "ens"]
            cost = [cost_of(g.pred_D1_min_cost.to_numpy(), g.y.to_numpy()) for _, g in s.groupby("seed")]
            auc = [roc_auc_score(g.y, g.score) for _, g in s.groupby("seed")]
            wf = [np.mean([roc_auc_score(h.y, h.score) for _, h in g.groupby("outer")]) for _, g in s.groupby("seed")]
            rows.append({"family": f, "draw": d, "variant": n, "cost_seed_mean": np.mean(cost),
                         "cost_seed_sd": np.std(cost, ddof=1), "ens_cost": ens_cost[n].mean(),
                         "roc_auc_seed_mean": np.mean(auc), "within_fold_roc_auc": np.mean(wf),
                         "selected_feature_sets": ";".join(sel[n][str(k)]["config"]["feature_set"] for k in range(SD.KO))})
    va = pd.DataFrame(rows)
    va["rank_in_draw"] = va.groupby("draw").ens_cost.rank(method="min").astype(int)      # E. ranks
    va.to_csv(OUT / "variants.csv", index=False)

    # B. draw-matched comparisons
    rows = []
    for d in (1, 2, 3):
        a = DRAWS["E-DDQN"][d - 1]
        for f in OTHERS:
            t = boot_test(ens_cost[a], ens_cost[DRAWS[f][d - 1]], grp)
            rows.append({"draw": d, "B": f, **t})
    dm = pd.DataFrame(rows)
    for d in (1, 2, 3):
        m = dm.draw == d
        dm.loc[m, "p_boot_holm"] = holm(dm.loc[m, "p_boot"])
        dm.loc[m, "p_signflip_holm"] = holm(dm.loc[m, "p_signflip"])
    dm.to_csv(OUT / "draw_matched.csv", index=False)

    # C. draw-averaged comparisons; B1 minimum detectable effect
    avg = {f: np.mean([ens_cost[n] for n in DRAWS[f]], axis=0) for f in FAMILIES}
    rows = []
    for f in OTHERS:
        t = boot_test(avg["E-DDQN"], avg[f], grp)
        rows.append({"B": f, "cost_eddqn": avg["E-DDQN"].mean(), "cost_B": avg[f].mean(), **t, "mde_cost": Z * t["boot_sd"]})
    da = pd.DataFrame(rows)
    da["p_boot_holm"], da["p_signflip_holm"] = holm(da.p_boot), holm(da.p_signflip)
    da.to_csv(OUT / "draw_averaged.csv", index=False)

    # D. every family vs always alarming (always alarm: cost 1 on every non-event second)
    always = (y == 0).astype(float)
    rows = [{"family": f, "cost": avg[f].mean(), "cost_always": always.mean(), **boot_test(avg[f], always, grp)} for f in FAMILIES]
    aa = pd.DataFrame(rows)
    aa["p_boot_holm"], aa["p_signflip_holm"] = holm(aa.p_boot), holm(aa.p_signflip)
    aa.to_csv(OUT / "vs_always_alarm.csv", index=False)

    # B1 (primary draw), B2 (seed-mean paired tests)
    rows = []
    for f in OTHERS:
        t = boot_test(ens_cost["E-DDQN"], ens_cost[f], grp)
        sm = boot_test(seed_cost_vec(P["E-DDQN"]), seed_cost_vec(P[f]), grp)
        rows.append({"B": f, "ens_diff": t["diff"], "ens_p_boot": t["p_boot"], "mde_cost": Z * t["boot_sd"],
                     "mde_auc": Z * auc_boot_sd(E["E-DDQN"].score.to_numpy(), E[f].score.to_numpy(), y, grp),
                     "seedmean_diff": sm["diff"], "seedmean_ci_low": sm["ci_low"], "seedmean_ci_high": sm["ci_high"],
                     "seedmean_p_boot": sm["p_boot"], "seedmean_p_signflip": sm["p_signflip"]})
    pm = pd.DataFrame(rows)
    pm["seedmean_p_boot_holm"], pm["seedmean_p_signflip_holm"] = holm(pm.seedmean_p_boot), holm(pm.seedmean_p_signflip)
    pm.to_csv(OUT / "primary_mde_and_seed_mean_tests.csv", index=False)

    # B3 ensemble vs seed mean (descriptive)
    rows = []
    for f in FAMILIES:
        p = P[f]
        s = p[p.seed != "ens"]
        rec = [g.pred_D1_min_cost[g.y == 1].mean() for _, g in s.groupby("seed")]
        fpr = [g.pred_D1_min_cost[g.y == 0].mean() for _, g in s.groupby("seed")]
        e = E[f]
        # D1 thresholds are chosen per outer fold: position of the ensemble threshold within the per-seed range
        pos, outside = [], 0
        for k in range(SD.KO):
            ts = s[s.outer == k].groupby("seed").thr_D1_min_cost.first()
            te = e[e.outer == k].thr_D1_min_cost.iloc[0]
            outside += int(te < ts.min() or te > ts.max())
            pos.append((te - ts.min()) / (ts.max() - ts.min()) if ts.max() > ts.min() else np.nan)
        rows.append({"family": f, "seed_mean_recall": np.mean(rec), "seed_mean_fpr": np.mean(fpr),
                     "ens_recall": e.pred_D1_min_cost[e.y == 1].mean(), "ens_fpr": e.pred_D1_min_cost[e.y == 0].mean(),
                     "score_sd_ratio_ens_to_seed": e.score.std() / np.mean([g.score.std() for _, g in s.groupby("seed")]),
                     "seed_mean_cost": np.mean([cost_of(g.pred_D1_min_cost.to_numpy(), g.y.to_numpy()) for _, g in s.groupby("seed")]),
                     "ens_cost": ens_cost[f].mean(), "ens_thr_position_in_seed_range": np.nanmean(pos),
                     "folds_ens_thr_outside_seed_range": outside})
    pd.DataFrame(rows).to_csv(OUT / "ensemble_vs_seed_mean.csv", index=False)

    # decision rules
    rf = da.set_index("B").loc["RF"]
    span = va.groupby("family").agg(cost_span=("cost_seed_mean", lambda x: x.max() - x.min()),
                                    wf_span=("within_fold_roc_auc", lambda x: x.max() - x.min()))
    dec = {"A1_rf_difference": "robust" if (rf.p_boot_holm < 0.05 and rf.p_signflip_holm < 0.05) else "draw-specific",
           "A2_eddqn_better_than": [r.B for r in da.itertuples() if r.diff < 0 and r.p_boot_holm < 0.05 and r.p_signflip_holm < 0.05],
           "A3_draw_dependent": {f: bool(r.cost_span > 0.05 or r.wf_span > 0.05) for f, r in span.iterrows()},
           "A3_spans": {f: {"cost": float(r.cost_span), "within_fold_auc": float(r.wf_span)} for f, r in span.iterrows()},
           "A4_beats_always_alarm": [r.family for r in aa.itertuples() if r.diff < 0 and r.p_boot_holm < 0.05 and r.p_signflip_holm < 0.05]}
    (OUT / "decisions.json").write_text(json.dumps(dec, indent=1))
    write_tables_and_macros(va, dm, da, aa, pm, pd.read_csv(OUT / "ensemble_vs_seed_mean.csv"), dec)
    print(json.dumps(dec, indent=1))


KEY = {"E-DDQN": "EDDQN", "DQN": "DQN", "E-CB": "ECB", "CS-MLP": "CSMLP", "LogReg": "LogReg", "HGB": "HGB", "RF": "RF"}
WORD = {1: "One", 2: "Two", 3: "Three"}


def write_tables_and_macros(va, dm, da, aa, pm, ev, dec):
    def f3(x):
        return f"{x:+.3f}".replace("-", "$-$")

    def pv(x):
        return "$<$0.001" if x < 0.001 else f"{x:.3f}"
    M = {}

    def put(k, val):
        assert k.isalpha(), k
        M["AD" + k] = val
    # table: every family, three draws
    body = ["\\begin{tabular}{llccccc}", "\\toprule",
            "Family & Draw & Cost$_{5:1}$ (seed mean) & Ens.\\ cost & ROC-AUC & Within-fold AUC & Rank \\\\", "\\midrule"]
    for f in FAMILIES:
        for r in va[va.family == f].itertuples():
            body.append(f"{f if r.draw == 1 else ''} & {r.draw}{' (primary)' if r.draw == 1 else ''} & "
                        f"{r.cost_seed_mean:.3f}$\\pm${r.cost_seed_sd:.3f} & {r.ens_cost:.3f} & {r.roc_auc_seed_mean:.3f} & "
                        f"{r.within_fold_roc_auc:.3f} & {r.rank_in_draw} \\\\")
        if f != FAMILIES[-1]:
            body.append("\\addlinespace[2pt]")
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_ad_variants.tex").write_text("\n".join(body) + "\n")
    # table: draw-averaged comparisons, seed-mean tests and MDE
    body = ["\\begin{tabular}{lcccccccc}", "\\toprule",
            "B & \\multicolumn{4}{c}{Draw-averaged: E-DDQN$-$B} & \\multicolumn{2}{c}{Primary seed means: E-DDQN$-$B} & \\multicolumn{2}{c}{MDE (primary)} \\\\",
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
            " & $\\Delta$Cost & 95\\,\\% CI & $p_{\\text{boot}}^{\\text{Holm}}$ & $p_{\\text{sign-flip}}^{\\text{Holm}}$ & $\\Delta$Cost & $p_{\\text{boot}}^{\\text{Holm}}$ / $p_{\\text{sign-flip}}^{\\text{Holm}}$ & Cost & ROC-AUC \\\\",
            "\\midrule"]
    pmi = pm.set_index("B")
    for r in da.itertuples():
        s = pmi.loc[r.B]
        body.append(f"{r.B} & {f3(r.diff)} & [{f3(r.ci_low)}, {f3(r.ci_high)}] & {pv(r.p_boot_holm)} & {pv(r.p_signflip_holm)} & "
                    f"{f3(s.seedmean_diff)} & {pv(s.seedmean_p_boot_holm)} / {pv(s.seedmean_p_signflip_holm)} & "
                    f"{s.mde_cost:.3f} & {s.mde_auc:.3f} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_ad_pairs.tex").write_text("\n".join(body) + "\n")
    # table: draw-averaged cost vs always alarming
    body = ["\\begin{tabular}{lccccc}", "\\toprule",
            "Family & Draw-averaged cost & $\\Delta$ vs always alarm & 95\\,\\% CI & $p_{\\text{boot}}^{\\text{Holm}}$ & $p_{\\text{sign-flip}}^{\\text{Holm}}$ \\\\",
            "\\midrule"]
    for r in aa.itertuples():
        body.append(f"{r.family} & {r.cost:.3f} & {f3(r.diff)} & [{f3(r.ci_low)}, {f3(r.ci_high)}] & {pv(r.p_boot_holm)} & {pv(r.p_signflip_holm)} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_ad_always.tex").write_text("\n".join(body) + "\n")
    # macros
    for f in FAMILIES:
        v, k = va[va.family == f], KEY[f]
        put(k + "CostMin", f"{v.cost_seed_mean.min():.3f}"); put(k + "CostMax", f"{v.cost_seed_mean.max():.3f}")
        put(k + "EnsMin", f"{v.ens_cost.min():.3f}"); put(k + "EnsMax", f"{v.ens_cost.max():.3f}")
        put(k + "WfMin", f"{v.within_fold_roc_auc.min():.3f}"); put(k + "WfMax", f"{v.within_fold_roc_auc.max():.3f}")
        put(k + "CostSpan", f"{dec['A3_spans'][f]['cost']:.3f}"); put(k + "WfSpan", f"{dec['A3_spans'][f]['within_fold_auc']:.3f}")
        for r in v.itertuples():
            put(k + "Rank" + WORD[r.draw], str(r.rank_in_draw))
    for r in da.itertuples():
        k = "Avg" + KEY[r.B]
        put(k + "Diff", f3(r.diff)); put(k + "Low", f3(r.ci_low)); put(k + "High", f3(r.ci_high))
        put(k + "PHolm", pv(r.p_boot_holm)); put(k + "SfHolm", pv(r.p_signflip_holm))
        put(k + "CostB", f"{r.cost_B:.3f}")
    put("AvgEDDQNCost", f"{da.cost_eddqn.iloc[0]:.3f}")
    put("AvgDiffMin", f3(da["diff"].min())); put("AvgDiffMax", f3(da["diff"].max()))
    put("AvgPHolmMin", pv(da.p_boot_holm.min())); put("AvgSfHolmMin", pv(da.p_signflip_holm.min()))
    for r in aa.itertuples():
        k = "Alw" + KEY[r.family]
        put(k + "Diff", f3(r.diff)); put(k + "PHolm", pv(r.p_boot_holm)); put(k + "SfHolm", pv(r.p_signflip_holm))
    put("AlwPHolmMin", pv(aa.p_boot_holm.min())); put("AlwSfHolmMin", pv(aa.p_signflip_holm.min()))
    put("MdeCostMin", f"{pm.mde_cost.min():.3f}"); put("MdeCostMax", f"{pm.mde_cost.max():.3f}")
    put("MdeAucMin", f"{pm.mde_auc.min():.3f}"); put("MdeAucMax", f"{pm.mde_auc.max():.3f}")
    put("MdeAvgCostMin", f"{da.mde_cost.min():.3f}"); put("MdeAvgCostMax", f"{da.mde_cost.max():.3f}")
    for r in pm.itertuples():
        k = "Sm" + KEY[r.B]
        put(k + "Diff", f3(r.seedmean_diff)); put(k + "PHolm", pv(r.seedmean_p_boot_holm)); put(k + "SfHolm", pv(r.seedmean_p_signflip_holm))
        put(k + "Low", f3(r.seedmean_ci_low)); put(k + "High", f3(r.seedmean_ci_high))
        put("Mde" + KEY[r.B] + "Cost", f"{r.mde_cost:.3f}")
    put("SmPHolmMin", pv(pm.seedmean_p_boot_holm.min())); put("SmSfHolmMin", pv(pm.seedmean_p_signflip_holm.min()))
    for r in ev.itertuples():
        k = "Ens" + KEY[r.family]
        put(k + "SeedRecall", f"{100 * r.seed_mean_recall:.1f}"); put(k + "SeedFPR", f"{100 * r.seed_mean_fpr:.1f}")
        put(k + "EnsRecall", f"{100 * r.ens_recall:.1f}"); put(k + "EnsFPR", f"{100 * r.ens_fpr:.1f}")
        put(k + "SdRatio", f"{r.score_sd_ratio_ens_to_seed:.2f}"); put(k + "Outside", str(int(r.folds_ens_thr_outside_seed_range)))
    put("SdRatioMin", f"{ev.score_sd_ratio_ens_to_seed.min():.2f}"); put("SdRatioMax", f"{ev.score_sd_ratio_ens_to_seed.max():.2f}")
    put("AOne", dec["A1_rf_difference"])
    put("ATwo", ", ".join(dec["A2_eddqn_better_than"]) or "none")
    dep = [f for f, b in dec["A3_draw_dependent"].items() if b]
    put("AThree", ", ".join(dep) or "none"); put("AThreeN", str(len(dep)))
    put("AFour", ", ".join(dec["A4_beats_always_alarm"]) or "none")
    (OUT / "ad_macros.json").write_text(json.dumps(M, indent=1))
    (OUT / "ad_macros.tex").write_text("% generated by scripts/19_all_family_draws.py -- do not edit\n" +
                                       "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyze"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="run only these variants (default: all new variants)")
    a = ap.parse_args()
    if a.stage == "run":
        SD.run(a.only or list(NEW), a.workers)
    else:
        analyze()
