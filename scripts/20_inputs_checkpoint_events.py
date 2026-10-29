"""Round-6 checks (configs/posthoc.json, key "review_round_6").

Specified after all other results were known and committed before any of
these runs or analyses:
  * exposure_only_all : the five remaining families re-tuned with the feature
    set fixed to "exposure" (E-DDQN-exposure and DQN-exposure are reused);
  * fixed_epoch       : the primary configurations of the neural models
    retrained with the last epoch instead of the validation checkpoint;
  * event_level       : alarm events with a refractory period and onsets as
    the unit of detection, from the saved seed-ensemble decisions.

Usage:
  python scripts/20_inputs_checkpoint_events.py run [--workers N]
  python scripts/20_inputs_checkpoint_events.py analyze
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
from eddqn.metrics import cost_of, min_cost_value  # noqa: E402


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # pool workers pickle functions by module name
    spec.loader.exec_module(mod)
    return mod


AD = _module("all_draws", "19_all_family_draws.py")
SD, AG, RP = AD.SD, AD.AG, AD.RP
SPEC = json.loads((CONFIGS / "posthoc.json").read_text())["review_round_6"]
EXPO = {v["name"]: {**v, "budget": RP.PROTOCOL["hpo_budget"][v["base"]]} for v in SPEC["exposure_only_all"]["variants"]}
LAST = {v["name"]: v["base"] for v in SPEC["fixed_epoch"]["variants"]}
FAMILIES = ["E-DDQN", "DQN", "E-CB", "CS-MLP", "LogReg", "HGB", "RF"]
OUT = RESULTS / "tables" / "round6"
SEL_PATH = RESULTS / "selected_configs_round6.json"

SD.VARIANTS.update(EXPO)
SD.SEL_PATH = SEL_PATH
SD.HPO_PATH = RESULTS / "hpo_results_round6.csv"
EXP.POSTHOC_BASE.update({n: v["base"] for n, v in EXPO.items()})
EXP.POSTHOC_BASE.update(LAST)


# ------------------------------------------------------------------ run
def run(workers):
    SD.run(list(EXPO), workers)                     # search, selection and final training, as in script 13
    psel = json.loads((RESULTS / "selected_configs.json").read_text())
    jobs = []
    for name, base in LAST.items():
        cfg = {k: {"config": {**psel[base][k]["config"], "checkpoint": "last"}} for k in psel[base]}
        jobs += RP.final_jobs(base, cfg, tag=name, score_attacks=False)
    RP.run_jobs(jobs, workers)


# -------------------------------------------------------------- helpers
def preds(name, F):
    path = AG.PRED / f"{name}.csv.gz"
    if not path.exists():
        AG.predictions_for(name, AG.collect(name, F), F)
    return pd.read_csv(path, dtype={"seed": str}).sort_values(["outer", "session_id", "step"]).reset_index(drop=True)


def threshold_optimism(name, F):
    """Seed mean of (pooled outer-test D1 cost - pooled-OOF minimum cost of the same final models)."""
    runs = AG.collect(name, F)
    p = preds(name, F)
    rows = []
    for s, g in p[p.seed != "ens"].groupby("seed"):
        test = cost_of(g.pred_D1_min_cost.to_numpy(), g.y.to_numpy())
        oof_cost = sum(min_cost_value(runs[(k, int(s))]["oof"].score.to_numpy(), runs[(k, int(s))]["oof"].y.to_numpy())
                       * len(runs[(k, int(s))]["oof"]) for k in range(SD.KO))
        oof_n = sum(len(runs[(k, int(s))]["oof"]) for k in range(SD.KO))
        rows.append({"seed": s, "test_cost": test, "oof_min_cost": oof_cost / oof_n})
    r = pd.DataFrame(rows)
    return {"test_cost": r.test_cost.mean(), "oof_min_cost": r.oof_min_cost.mean(),
            "optimism": (r.test_cost - r.oof_min_cost).mean()}


def events(sess, R):
    """Event-level quantities of one session (rows in temporal order)."""
    y, a, t = sess.y.to_numpy(), sess.pred.to_numpy(), sess.step.to_numpy()
    ev, last = [], -np.inf
    for ti, ai in zip(t, a):
        if ai == 1 and ti - last >= R:
            ev.append(ti)
            last = ti
    ev = np.asarray(ev)
    on = t[y == 1].min() if (y == 1).any() else None
    yt = dict(zip(t, y))
    if on is not None:
        win = ev[(ev >= on - 60) & (ev <= on + 30)]
        detected, lead = len(win) > 0, (on - win.min()) if len(win) else np.nan
        fa = [e for e in ev if yt[e] == 0 and not (on - 60 <= e <= on)]
    else:
        detected, lead = None, np.nan
        fa = [e for e in ev if yt[e] == 0]
    return {"has_onset": on is not None, "detected": detected, "lead": lead, "false_alarms": len(fa),
            "nonevent_seconds": int((y == 0).sum()),
            "event_cost": (5 * (on is not None and not detected)) + len(fa)}


def event_table(dec, R):
    rows = [{"session_id": s, "group_id": g.group_id.iloc[0], **events(g, R)} for s, g in dec.groupby("session_id")]
    return pd.DataFrame(rows)


def boot_mean(v, grp, seed=7, nb=2000):
    """Group bootstrap of the mean over sessions."""
    uniq = np.unique(grp)
    idx_of = {g: np.flatnonzero(grp == g) for g in uniq}
    rng = np.random.default_rng(seed)
    reps = [v[np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])].mean() for _ in range(nb)]
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


# -------------------------------------------------------------- analyze
def analyze():
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    F = AG.Frames()
    P = {n: preds(n, F) for n in FAMILIES + ["Time-LR"]}
    ref = P["E-DDQN"]
    E = {n: SD.ens(p) for n, p in P.items()}
    y, grp = E["E-DDQN"].y.to_numpy(), E["E-DDQN"].group_id.to_numpy()

    # ---------------- exposure-only, every family
    xn = {f: f"{f}-exposure" for f in FAMILIES}
    X = {f: preds(xn[f], F) for f in FAMILIES}
    for f in FAMILIES:
        assert (X[f][["session_id", "step"]].values == ref[["session_id", "step"]].values).all(), f
    XE = {f: SD.ens(X[f]) for f in FAMILIES}
    rows = []
    for f in FAMILIES:
        s = X[f][X[f].seed != "ens"]
        rows.append({"family": f, "cost_seed_mean": np.mean([cost_of(g.pred_D1_min_cost.to_numpy(), g.y.to_numpy()) for _, g in s.groupby("seed")]),
                     "ens_cost": cost_of(XE[f].pred_D1_min_cost.to_numpy(), y),
                     "within_fold_roc_auc": np.mean([np.mean([roc_auc_score(h.y, h.score) for _, h in g.groupby("outer")]) for _, g in s.groupby("seed")])})
    xo = pd.DataFrame(rows)
    cost_vec = {f: SD.cvec(XE[f].pred_D1_min_cost.to_numpy(), y) for f in FAMILIES}
    pr = []
    for f in FAMILIES[1:]:
        t = AD.boot_test(cost_vec["E-DDQN"], cost_vec[f], grp)
        au = SD.paired_auc_diff(XE["E-DDQN"].score.to_numpy(), XE[f].score.to_numpy(), y, grp, AD.NB, seed=6)
        pr.append({"B": f, **t, "auc_diff": au["diff"], "auc_ci_low": au["ci_low"], "auc_ci_high": au["ci_high"], "auc_p": au["p"]})
    xp = pd.DataFrame(pr)
    xp["p_boot_holm"], xp["p_signflip_holm"] = AD.holm(xp.p_boot), AD.holm(xp.p_signflip)
    # added without prior commitment: every exposure-only family against always alarming (Holm over seven)
    always = (y == 0).astype(float)
    xa = pd.DataFrame([{"family": f, **AD.boot_test(cost_vec[f], always, grp)} for f in FAMILIES])
    xa["p_boot_holm"], xa["p_signflip_holm"] = AD.holm(xa.p_boot), AD.holm(xa.p_signflip)
    xa.to_csv(OUT / "exposure_only_vs_always_alarm.csv", index=False)
    xo.to_csv(OUT / "exposure_only_families.csv", index=False)
    xp.to_csv(OUT / "exposure_only_pairs.csv", index=False)
    xi = xo.set_index("family")
    med = float(xi.drop("E-DDQN").within_fold_roc_auc.median())
    gap = med - float(xi.loc["E-DDQN", "within_fold_roc_auc"])
    f1 = "explained" if abs(gap) <= 0.03 else ("not explained" if gap > 0.05 else "inconclusive")   # spec F1, literally
    f2 = [r.B for r in xp.itertuples() if r.p_boot_holm < 0.05 and r.p_signflip_holm < 0.05]

    # ---------------- fixed epoch
    rows = []
    for name, base in LAST.items():
        b, l = threshold_optimism(base, F), threshold_optimism(name, F)
        eb, el = SD.ens(P[base]), SD.ens(preds(name, F))
        t = AD.boot_test(SD.cvec(el.pred_D1_min_cost.to_numpy(), y), SD.cvec(eb.pred_D1_min_cost.to_numpy(), y), grp)
        rows.append({"family": base, "cost_checkpoint": b["test_cost"], "cost_last": l["test_cost"],
                     "optimism_checkpoint": b["optimism"], "optimism_last": l["optimism"],
                     "oof_checkpoint": b["oof_min_cost"], "oof_last": l["oof_min_cost"],
                     "ens_diff_last_minus_checkpoint": t["diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"],
                     "p_boot": t["p_boot"], "p_signflip": t["p_signflip"]})
    fe = pd.DataFrame(rows)
    fe.to_csv(OUT / "fixed_epoch.csv", index=False)
    e = fe.set_index("family").loc["E-DDQN"]
    c1 = "substantial" if e.optimism_last < 0.5 * e.optimism_checkpoint else "not substantial"

    # ---------------- event level
    decs = {n: E[n][["session_id", "group_id", "step", "y"]].assign(pred=E[n].pred_D1_min_cost.to_numpy()) for n in FAMILIES + ["Time-LR"]}
    decs["Always alarm"] = E["E-DDQN"][["session_id", "group_id", "step", "y"]].assign(pred=1)
    rows, tabs = [], {}
    for R in (30, 60, 120):
        for n, d in decs.items():
            tb = event_table(d, R)
            tabs[(n, R)] = tb
            on = tb[tb.has_onset]
            lo, hi = boot_mean(tb.event_cost.to_numpy(float), tb.group_id.to_numpy())
            rows.append({"model": n, "R": R, "onsets": len(on), "detection_rate": on.detected.mean(),
                         "median_lead_s": float(np.nanmedian(on.lead)) if on.detected.any() else np.nan,
                         "false_alarms_per_10min": tb.false_alarms.sum() / (tb.nonevent_seconds.sum() / 600),
                         "event_cost": tb.event_cost.mean(), "event_cost_ci_low": lo, "event_cost_ci_high": hi})
    ev = pd.DataFrame(rows)
    ev.to_csv(OUT / "event_level.csv", index=False)
    pr = []
    for f in FAMILIES[1:]:
        a, b = tabs[("E-DDQN", 60)], tabs[(f, 60)]
        assert (a.session_id.values == b.session_id.values).all()
        g = a.group_id.to_numpy()
        t = AD.boot_test(a.event_cost.to_numpy(float), b.event_cost.to_numpy(float), g, seed=7)
        pr.append({"B": f, **t})
    ep = pd.DataFrame(pr)
    ep["p_boot_holm"], ep["p_signflip_holm"] = AD.holm(ep.p_boot), AD.holm(ep.p_signflip)
    ep.to_csv(OUT / "event_level_pairs.csv", index=False)
    e60 = ev[ev.R == 60]
    e1 = [r.model for r in e60.itertuples() if r.detection_rate >= 0.8 and r.false_alarms_per_10min <= 1.0]

    # ---------------- training prevalence (descriptive restatement of scripts/14 output)
    ec = pd.read_csv(RESULTS / "tables" / "score_scale" / "effective_cutoffs.csv")
    prev = {"split_min": float(ec.train_prevalence_min.min()), "split_max": float(ec.train_prevalence_max.max()),
            "fold_mean_min": float(ec.train_prevalence_mean.min()), "fold_mean_max": float(ec.train_prevalence_mean.max())}

    dec = {"F1_inputs_explain_deficit": f1, "F1_gap_median_minus_eddqn": gap, "F1_median_other": med,
           "F2_established_cost_differences": f2, "C1_checkpoint_explains_optimism": c1,
           "C1_eddqn_optimism_checkpoint": float(e.optimism_checkpoint), "C1_eddqn_optimism_last": float(e.optimism_last),
           "E1_useful_operating_point": e1, "prevalence": prev}
    (OUT / "decisions.json").write_text(json.dumps(dec, indent=1))
    write_macros(xo, xp, fe, ev, ep, dec, xa)
    print(json.dumps(dec, indent=1))


KEY = AD.KEY | {"Time-LR": "TimeLR", "Always alarm": "Always"}


def write_macros(xo, xp, fe, ev, ep, dec, xa):
    def f3(x):
        return f"{x:+.3f}".replace("-", "$-$")

    def pv(x):
        return "$<$0.001" if x < 0.001 else f"{x:.3f}"
    M = {}

    def put(k, v):
        assert k.isalpha(), k
        M["RS" + k] = v
    for r in xo.itertuples():
        put("Exp" + KEY[r.family] + "Wf", f"{r.within_fold_roc_auc:.3f}"); put("Exp" + KEY[r.family] + "Ens", f"{r.ens_cost:.3f}")
    put("ExpWfMin", f"{xo.within_fold_roc_auc.min():.3f}"); put("ExpWfMax", f"{xo.within_fold_roc_auc.max():.3f}")
    put("ExpMedianOther", f"{dec['F1_median_other']:.3f}"); put("ExpGap", f"{dec['F1_gap_median_minus_eddqn']:+.3f}".replace("-", "$-$"))
    put("FOne", dec["F1_inputs_explain_deficit"]); put("FTwo", ", ".join(dec["F2_established_cost_differences"]) or "none")
    for r in xp.itertuples():
        k = "ExpPair" + KEY[r.B]
        put(k + "Diff", f3(r.diff)); put(k + "PHolm", pv(r.p_boot_holm)); put(k + "SfHolm", pv(r.p_signflip_holm))
    put("ExpDiffMin", f3(xp["diff"].min())); put("ExpDiffMax", f3(xp["diff"].max()))
    put("ExpPHolmMin", pv(xp.p_boot_holm.min())); put("ExpSfHolmMin", pv(xp.p_signflip_holm.min()))
    for r in fe.itertuples():
        k = "Last" + KEY[r.family]
        put(k + "Cost", f"{r.cost_last:.3f}"); put(k + "CkCost", f"{r.cost_checkpoint:.3f}")
        put(k + "Opt", f3(r.optimism_last)); put(k + "CkOpt", f3(r.optimism_checkpoint))
        put(k + "Diff", f3(r.ens_diff_last_minus_checkpoint)); put(k + "P", pv(r.p_boot)); put(k + "SfP", pv(r.p_signflip))
    put("COne", dec["C1_checkpoint_explains_optimism"])
    for r in ev.itertuples():
        k = "Ev" + KEY[r.model] + {30: "Thirty", 60: "Sixty", 120: "OneTwenty"}[r.R]
        put(k + "Det", f"{100 * r.detection_rate:.0f}"); put(k + "Fa", f"{r.false_alarms_per_10min:.2f}")
        put(k + "Cost", f"{r.event_cost:.2f}")
        put(k + "Lead", "--" if np.isnan(r.median_lead_s) else f"{r.median_lead_s:.0f}")
    e60 = ev[ev.R == 60]
    put("EvOnsets", str(int(e60.onsets.iloc[0])))
    put("EvDetMin", f"{100 * e60.detection_rate.min():.0f}"); put("EvDetMax", f"{100 * e60.detection_rate.max():.0f}")
    prim = e60[~e60.model.isin(["Always alarm"])]
    put("EvFaMin", f"{prim.false_alarms_per_10min.min():.2f}"); put("EvFaMax", f"{prim.false_alarms_per_10min.max():.2f}")
    put("EOne", ", ".join(dec["E1_useful_operating_point"]) or "none")
    for r in ep.itertuples():
        k = "EvPair" + KEY[r.B]
        put(k + "Diff", f"{r.diff:+.2f}".replace("-", "$-$")); put(k + "PHolm", pv(r.p_boot_holm)); put(k + "SfHolm", pv(r.p_signflip_holm))
    put("EvPHolmMin", pv(ep.p_boot_holm.min())); put("EvSfHolmMin", pv(ep.p_signflip_holm.min()))
    for r in xa.itertuples():
        k = "ExpAlw" + KEY[r.family]
        put(k + "Diff", f3(r.diff)); put(k + "PHolm", pv(r.p_boot_holm)); put(k + "SfHolm", pv(r.p_signflip_holm))
    put("ExpAlwPHolmMin", pv(xa.p_boot_holm.min())); put("ExpAlwSfHolmMin", pv(xa.p_signflip_holm.min()))
    put("ExpAlwBelow", str(int((xa["diff"] < 0).sum())))
    # cost-curve range at R = 1.5-2 at three decimals (scripts/11 quotes it at two), all models except E-DDQN
    cc = pd.read_csv(RESULTS / "tables" / "posthoc" / "cost_curves.csv")
    low = cc[cc.ratio.isin([1.5, 2.0]) & (cc.model != "E-DDQN")].relative_cost
    put("CcLowMin", f"{low.min():.3f}"); put("CcLowMax", f"{low.max():.3f}")
    pr = dec["prevalence"]
    put("PrevSplitMin", f"{pr['split_min']:.3f}"); put("PrevSplitMax", f"{pr['split_max']:.3f}")
    put("PrevFoldMin", f"{pr['fold_mean_min']:.3f}"); put("PrevFoldMax", f"{pr['fold_mean_max']:.3f}")
    (OUT / "rs_macros.json").write_text(json.dumps(M, indent=1))
    (OUT / "rs_macros.tex").write_text("% generated by scripts/20_inputs_checkpoint_events.py -- do not edit\n" +
                                       "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
    # supplement tables
    lab = {30: "30", 60: "60", 120: "120"}
    body = ["\\begin{tabular}{lcccccc}", "\\toprule", "Family & Ens.\\ cost & Within-fold AUC & $\\Delta$Cost (E-DDQN$-$B) & $p_{\\text{boot}}^{\\text{Holm}}$ / $p_{\\text{sign-flip}}^{\\text{Holm}}$ & $\\Delta$ vs always alarm & $p_{\\text{boot}}^{\\text{Holm}}$ / $p_{\\text{sign-flip}}^{\\text{Holm}}$ \\\\", "\\midrule"]
    xpi, xai = xp.set_index("B"), xa.set_index("family")
    for r in xo.itertuples():
        a = xai.loc[r.family]
        alw = f"{f3(a['diff'])} & {pv(a.p_boot_holm)} / {pv(a.p_signflip_holm)}"
        if r.family == "E-DDQN":
            body.append(f"E-DDQN & {r.ens_cost:.3f} & {r.within_fold_roc_auc:.3f} & -- & -- & {alw} \\\\")
        else:
            q = xpi.loc[r.family]
            body.append(f"{r.family} & {r.ens_cost:.3f} & {r.within_fold_roc_auc:.3f} & {f3(q['diff'])} & {pv(q.p_boot_holm)} / {pv(q.p_signflip_holm)} & {alw} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_rs_exposure.tex").write_text("\n".join(body) + "\n")
    body = ["\\begin{tabular}{lcccccc}", "\\toprule", "Family & \\multicolumn{2}{c}{Test cost (seed mean)} & \\multicolumn{2}{c}{Threshold optimism} & $\\Delta$Cost last$-$checkpoint & $p_{\\text{boot}}$ / $p_{\\text{sign-flip}}$ \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", " & checkpoint & last epoch & checkpoint & last epoch & (ensembles) & \\\\", "\\midrule"]
    for r in fe.itertuples():
        body.append(f"{r.family} & {r.cost_checkpoint:.3f} & {r.cost_last:.3f} & {f3(r.optimism_checkpoint)} & {f3(r.optimism_last)} & {f3(r.ens_diff_last_minus_checkpoint)} & {pv(r.p_boot)} / {pv(r.p_signflip)} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_rs_lastepoch.tex").write_text("\n".join(body) + "\n")
    body = ["\\begin{tabular}{llccccc}", "\\toprule", "$R$ (s) & Model & Onsets detected (\\%) & Median lead (s) & False-alarm events / 10 min & Event cost & 95\\,\\% CI \\\\", "\\midrule"]
    for R in (30, 60, 120):
        for r in ev[ev.R == R].itertuples():
            lead = "--" if np.isnan(r.median_lead_s) else f"{r.median_lead_s:.0f}"
            body.append(f"{lab[R] if r.model == 'E-DDQN' else ''} & {r.model} & {100 * r.detection_rate:.0f} & {lead} & {r.false_alarms_per_10min:.2f} & {r.event_cost:.2f} & [{r.event_cost_ci_low:.2f}, {r.event_cost_ci_high:.2f}] \\\\")
        if R != 120:
            body.append("\\addlinespace[2pt]")
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "tab_rs_events.tex").write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyze"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.workers) if a.stage == "run" else analyze()
