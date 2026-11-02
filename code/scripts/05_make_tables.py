"""Write LaTeX macros and tables for the manuscript and supplement.

Single source of truth: results/tables/*.csv (from 04_aggregate.py),
results/preprocessing/*.csv, results/audit/*.csv, results/original_repro/*.
No number is typed by hand in the manuscript: every result appears through a
macro defined in results/tables/latex/numbers.tex or a generated table.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn import CONFIGS, RESULTS, REVISION_ROOT  # noqa: E402
from eddqn.features import display_name  # noqa: E402

TAB = RESULTS / "tables"
GEN = RESULTS / "tables" / "latex"
SGEN = RESULTS / "tables" / "latex_supplement"
for d in (GEN, SGEN):
    d.mkdir(parents=True, exist_ok=True)
PROTOCOL = json.loads((CONFIGS / "protocol.json").read_text())

MODELS = ["E-DDQN", "E-CB", "DQN", "CS-MLP", "LogReg", "HGB", "RF"]
MKEY = {"E-DDQN": "EDDQN", "E-CB": "ECB", "DQN": "DQN", "CS-MLP": "CSMLP", "LogReg": "LR", "HGB": "HGB", "RF": "RF"}
RULES = {"D1_min_cost": "Cost", "D0_bayes_1_6": "Bayes", "recall>=0.95": "Rec"}
METRICS = {"recall": "Recall", "specificity": "Spec", "precision": "Prec", "accuracy": "Acc", "f1": "Fone",
           "balanced_accuracy": "BalAcc", "mcc": "MCC", "g_mean": "Gmean", "fpr": "FPR", "fnr": "FNR",
           "expected_cost": "Cost", "roc_auc": "RocAuc", "pr_auc": "PrAuc", "FN": "FN", "FP": "FP",
           "TP": "TP", "TN": "TN", "alarm_rate": "Alarm"}
PCT = {"recall", "specificity", "precision", "accuracy", "f1", "balanced_accuracy", "g_mean", "fpr", "fnr", "alarm_rate"}
macros: dict[str, str] = {}


def mname(*parts):
    s = "".join(parts)
    s = re.sub(r"[^A-Za-z]", "", s.replace("0", "Zero").replace("1", "One").replace("2", "Two")
               .replace("3", "Three").replace("4", "Four").replace("5", "Five").replace("6", "Six")
               .replace("7", "Seven").replace("8", "Eight").replace("9", "Nine"))
    return s


def fmt(metric, v, sd=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if metric in PCT:
        return f"{100 * v:.1f}"
    if metric in ("FN", "FP", "TP", "TN"):
        return f"{v:.1f}" if sd else f"{int(round(v)):,}"
    return f"{v:.3f}"


def put(name, value):
    macros[name] = value


# ------------------------------------------------------------------ data
def data_macros():
    roles = pd.read_csv(RESULTS / "preprocessing" / "counts_by_role.csv").set_index("role")
    games = pd.read_csv(RESULTS / "preprocessing" / "counts_eval_by_game.csv").set_index("game")
    folds = pd.read_csv(REVISION_ROOT / "data" / "processed" / "outer_folds.csv")
    put("nSessionsEval", f"{int(roles.loc['evaluation', 'sessions'])}")
    put("nSamplesEval", f"{int(roles.loc['evaluation', 'samples']):,}")
    put("nPosEval", f"{int(roles.loc['evaluation', 'positives']):,}")
    put("nSessionsDev", f"{int(roles.loc['development', 'sessions'])}")
    put("nGroupsEval", f"{folds.group_id.nunique()}")
    put("nOuter", f"{PROTOCOL['k_outer']}")
    put("nInner", f"{PROTOCOL['k_inner']}")
    put("prevEval", f"{100 * roles.loc['evaluation', 'prevalence']:.1f}\\,\\%")
    put("prevFlight", f"{100 * games.loc['Flight', 'prevalence']:.1f}\\,\\%")
    put("prevRace", f"{100 * games.loc['Race', 'prevalence']:.1f}\\,\\%")
    q = json.loads((RESULTS / "preprocessing" / "data_quality.json").read_text())["evaluation"]
    put("nNeverPos", f"{q['sessions_never_positive']}")
    put("onsetMedian", f"{q['onset_time_s']['median']:.0f}")


# ------------------------------------------------------------- leakage
def leakage_macros_and_table():
    demo = pd.read_csv(RESULTS / "audit" / "leakage_demo.csv")
    dup = json.loads((RESULTS / "audit" / "leakage_demo_duplicates.json").read_text())
    lab = {"A": "Old", "B": "Row", "C": "Grp"}
    for _, r in demo.iterrows():
        k = lab[r.protocol[0]]
        for m in ("accuracy", "recall", "specificity", "mcc", "roc_auc"):
            put(mname("Leak", k, METRICS[m]), fmt(m, r[m]))
    put("LeakDupPos", f"{dup['test_positive_rows_with_exact_copy_in_train']:,}")
    put("LeakTestPos", f"{dup['test_positive_rows']:,}")
    put("LeakDupPct", f"{100 * dup['test_positive_rows_with_exact_copy_in_train'] / dup['test_positive_rows']:.0f}")
    rows = []
    names = {"A": "Oversample to 5,243/grade, then 70/15/15 row split (submitted protocol)",
             "B": "Row-level random split, no oversampling",
             "C": "Grouped 5-fold split (session-linkage groups)"}
    for _, r in demo.iterrows():
        rows.append(f"HGB & {names[r.protocol[0]]} & {fmt('accuracy', r.accuracy)} & {fmt('recall', r.recall)} & "
                    f"{fmt('specificity', r.specificity)} & {fmt('mcc', r.mcc)} & {fmt('roc_auc', r.roc_auc)} \\\\")
    for v, nm in (("leaky_oversampled", "Submitted script, submitted protocol (proxy)"),
                  ("grouped", "Submitted script, grouped split (proxy)")):
        p = RESULTS / "original_repro" / v / "split_comparison.csv"
        if p.exists():
            t = pd.read_csv(p).set_index("Split").loc["Test"]
            rows.append(f"E-DDQN (submitted code) & {nm} & {100 * t.Accuracy:.1f} & {100 * t.Recall:.1f} & "
                        f"{100 * t.Specificity:.1f} & {t.MCC:.3f} & -- \\\\")
            key = "Old" if v.startswith("leaky") else "Grp"
            put(mname("Orig", key, "Recall"), f"{100 * t.Recall:.1f}")
            put(mname("Orig", key, "Spec"), f"{100 * t.Specificity:.1f}")
            put(mname("Orig", key, "Acc"), f"{100 * t.Accuracy:.1f}")
            put(mname("Orig", key, "MCC"), f"{t.MCC:.3f}")
    (GEN / "tab_leakage.tex").write_text(
        "\\begin{tabular}{llccccc}\n\\toprule\nModel & Evaluation protocol & Acc. & Recall & Spec. & MCC & ROC-AUC \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# --------------------------------------------------------------- main results
def s_d1_fpr(summ):
    return summ[(summ.rule == "D1_min_cost") & summ.model.isin(MODELS)]["fpr_mean"]


def main_macros_and_tables():
    summ = pd.read_csv(TAB / "metrics_summary.csv")
    triv = pd.read_csv(TAB / "trivial_baselines.csv")
    for _, r in summ.iterrows():
        if r.model not in MKEY or r.rule not in RULES:
            continue
        for m, mk in METRICS.items():
            base = mname(MKEY[r.model], RULES[r.rule], mk)
            put(base + "Mean", fmt(m, r[f"{m}_mean"]))
            put(base + "SD", fmt(m, r[f"{m}_std"], sd=True))
            put(base + "Ens", fmt(m, r[f"ens_{m}"]))
    for _, r in triv.iterrows():
        k = "Always" if r.model.startswith("Always") else "Never"
        for m, mk in METRICS.items():
            if m in r and not pd.isna(r[m]):
                put(mname(k, mk), fmt(m, r[m]))
    cols = [("recall", "Recall"), ("specificity", "Spec."), ("precision", "Prec."), ("f1", "F1"),
            ("balanced_accuracy", "Bal. acc."), ("mcc", "MCC"), ("expected_cost", "Cost$_{5:1}$"),
            ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]

    def table(rule, fname, caption_rows=True):
        lines = []
        s = summ[summ.rule == rule].set_index("model")
        best_cost = s["expected_cost_mean"].min()
        for mdl in MODELS:
            if mdl not in s.index:
                continue
            r = s.loc[mdl]
            cells = []
            for m, _ in cols:
                val = f"{fmt(m, r[f'{m}_mean'])}$\\pm${fmt(m, r[f'{m}_std'], sd=True)}"
                if m == "expected_cost" and abs(r[f"{m}_mean"] - best_cost) < 1e-12:
                    val = f"\\textbf{{{val}}}"
                cells.append(val)
            name = f"\\textbf{{{mdl}}}" if mdl == "E-DDQN" else mdl
            lines.append(f"{name} & " + " & ".join(cells) + f" & {fmt('FN', r['ens_FN'])} & {fmt('FP', r['ens_FP'])} \\\\")
        lines.append("\\midrule")
        for _, r in triv.iterrows():
            cells = [fmt(m, r[m]) if m in r and not pd.isna(r[m]) else "--" for m, _ in cols]
            lines.append(f"{r.model} & " + " & ".join(cells) + f" & {fmt('FN', r['FN'])} & {fmt('FP', r['FP'])} \\\\")
        head = " & ".join(h for _, h in cols)
        (GEN / fname).write_text("\\begin{tabular}{l" + "c" * (len(cols) + 2) + "}\n\\toprule\nModel & " + head +
                                 " & FN & FP \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    table("D1_min_cost", "tab_main_D1.tex")
    table("D0_bayes_1_6", "tab_main_D0.tex")
    table("recall>=0.95", "tab_main_R95.tex")
    fprs = 100 * s_d1_fpr(summ)
    put("DOneFPRMin", f"{fprs.min():.0f}")
    put("DOneFPRMax", f"{fprs.max():.0f}")
    # best baseline by cost (D1)
    s = summ[summ.rule == "D1_min_cost"].set_index("model")
    ranking = s["expected_cost_mean"].sort_values()
    put("BestModelCost", ranking.index[0])
    put("BestBaseline", [m for m in ranking.index if m != "E-DDQN"][0])
    put("EDDQNCostRank", str(list(ranking.index).index("E-DDQN") + 1) if "E-DDQN" in ranking.index else "--")
    roc = s["roc_auc_mean"].sort_values(ascending=False)
    put("BestModelAUC", roc.index[0])
    put("EDDQNAUCRank", str(list(roc.index).index("E-DDQN") + 1) if "E-DDQN" in roc.index else "--")
    put("nModels", str(len(s)))


def ci_macros():
    p = TAB / "bootstrap_cis_D1.json"
    if not p.exists():
        return
    cis = json.loads(p.read_text())
    lines = []
    for mdl in MODELS:
        if mdl not in cis:
            continue
        c = cis[mdl]
        for m in ("recall", "specificity", "expected_cost", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"):
            if m in c:
                base = mname(MKEY[mdl], "CI", METRICS[m])
                put(base + "Lo", fmt(m, c[m]["ci_low"]))
                put(base + "Hi", fmt(m, c[m]["ci_high"]))
        lines.append(f"{mdl} & " + " & ".join(
            f"{fmt(m, c[m]['estimate'])} [{fmt(m, c[m]['ci_low'])}, {fmt(m, c[m]['ci_high'])}]"
            for m in ("recall", "specificity", "expected_cost", "roc_auc", "pr_auc")) + " \\\\")
    (GEN / "tab_ci.tex").write_text("\\begin{tabular}{lccccc}\n\\toprule\nModel (seed ensemble, D1) & Recall & Spec. & Cost$_{5:1}$ & ROC-AUC & PR-AUC \\\\\n\\midrule\n" +
                                    "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def paired_tables():
    for rule, tag in (("D1_min_cost", "D1"), ("D0_bayes_1_6", "D0"), ("recall>=0.95", "R95")):
        p = TAB / f"paired_tests_{rule.replace('>=', 'ge')}.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        lines = []
        for _, r in t.iterrows():
            def p3(x):
                return f"{x:.2g}" if x >= 1e-3 else "$<$0.001"
            lines.append(f"{r.B} & {r.b_A_right_B_wrong:,} & {r.c_A_wrong_B_right:,} & {r.errors_A:,} & {r.errors_B:,} & "
                         f"{r.chi2_cc:.1f} & {p3(r.p_exact)} & {r.cost_A_minus_B:+.3f} [{r.cost_diff_ci_low:+.3f}, {r.cost_diff_ci_high:+.3f}] & "
                         f"{p3(r.cost_diff_p_cluster_bootstrap)} \\\\")
            k = mname("Pair", tag, MKEY[r.B])
            put(k + "CostDiff", f"{r.cost_A_minus_B:+.3f}")
            put(k + "CostLo", f"{r.cost_diff_ci_low:+.3f}")
            put(k + "CostHi", f"{r.cost_diff_ci_high:+.3f}")
            put(k + "Pboot", p3(r.cost_diff_p_cluster_bootstrap))
            put(k + "Pmcn", p3(r.p_exact))
            put(k + "b", f"{r.b_A_right_B_wrong:,}")
            put(k + "c", f"{r.c_A_wrong_B_right:,}")
        (GEN / f"tab_paired_{tag}.tex").write_text(
            "\\begin{tabular}{lcccccccc}\n\\toprule\nB (vs. A = E-DDQN) & $b$ & $c$ & err$_A$ & err$_B$ & $\\chi^2_{cc}$ & $p_{\\text{exact}}$ & "
            "$\\Delta$Cost$_{5:1}$ (A$-$B) [95\\% CI] & $p_{\\text{group}}$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def matched_tables():
    t = pd.read_csv(TAB / "matched_operating_points.csv")
    tgt = ["recall>=0.9", "recall>=0.95", "spec>=0.8"]
    lines = []
    for mdl in MODELS:
        g = t[t.model == mdl].set_index("target (on validation OOF)")
        if g.empty:
            continue
        cells = []
        for k in tgt:
            if k in g.index:
                r = g.loc[k]
                cells += [fmt("recall", r.test_recall), fmt("fpr", r.test_fpr), f"{int(r.FP):,}"]
            else:
                cells += ["--"] * 3
        lines.append(f"{mdl} & " + " & ".join(cells) + " \\\\")
    (GEN / "tab_matched.tex").write_text(
        "\\begin{tabular}{l" + "ccc" * len(tgt) + "}\n\\toprule\n & \\multicolumn{3}{c}{val. recall $\\ge$ 0.90} & "
        "\\multicolumn{3}{c}{val. recall $\\ge$ 0.95} & \\multicolumn{3}{c}{val. specificity $\\ge$ 0.80} \\\\\n"
        "Model & Rec. & FPR & FP & Rec. & FPR & FP & Rec. & FPR & FP \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    o = pd.read_csv(TAB / "roc_points_at_recall_targets_descriptive.csv")
    for _, r in o.iterrows():
        if r.model in MKEY:
            put(mname(MKEY[r.model], "RocFPRat", str(int(round(100 * r.recall_target)))), f"{100 * r.fpr_at_target_on_roc:.1f}")


def event_table():
    t = pd.read_csv(TAB / "event_level_D1.csv")
    lines = []
    for _, r in t.iterrows():
        lines.append(f"{r.model} & {r.events_detected}/{r.sessions_with_event} & {r.median_detection_delay_s:.0f} & "
                     f"{100 * r.mean_pre_onset_alarm_fraction:.1f} & {100 * r.mean_alarm_fraction_in_event_free_sessions:.1f} \\\\")
        if r.model in MKEY:
            k = MKEY[r.model]
            put(mname(k, "EvDetected"), f"{r.events_detected}")
            put(mname(k, "EvDelay"), f"{r.median_detection_delay_s:.0f}")
            put(mname(k, "EvFreeAlarm"), f"{100 * r.mean_alarm_fraction_in_event_free_sessions:.1f}")
            put(mname(k, "EvPreAlarm"), f"{100 * r.mean_pre_onset_alarm_fraction:.1f}")
    put("nEventSessions", f"{int(t.sessions_with_event.iloc[0])}")
    put("nEventFreeSessions", f"{int(t.sessions_without_event.iloc[0])}")
    (GEN / "tab_event.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\nModel & Events detected & Median delay (s) & Alarm \\% before onset & Alarm \\% in event-free sessions \\\\\n\\midrule\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def selected_config_table():
    sel = json.loads((RESULTS / "selected_configs.json").read_text())
    lines = []
    for mdl in MODELS:
        if mdl not in sel:
            continue
        for k, v in sorted(sel[mdl].items()):
            c = v["config"]
            keys = [x for x in ("feature_set", "hidden", "gamma", "full_info", "rho", "tau", "hard_every", "norm", "dropout",
                                "lr", "weight_decay", "batch_size", "eps_decay", "warmup", "updates_per_learn", "sampling",
                                "C", "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization",
                                "max_depth", "max_features") if x in c]
            desc = ", ".join(f"{x}={c[x]}" for x in keys).replace("_", "\\_")
            lines.append(f"{mdl} & {k} & {v['pooled_oof_min_cost']:.3f} & \\footnotesize {desc} \\\\")
    (SGEN / "tab_selected_configs.tex").write_text(
        "\\begin{longtable}{llcp{0.55\\textwidth}}\n\\toprule\nModel & Outer & OOF cost & Selected configuration \\\\\n\\midrule\n\\endhead\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n")
    e = sel.get("E-DDQN", {})
    gam = [e[k]["config"].get("gamma") for k in sorted(e)]
    put("EDDQNGammas", ", ".join(f"{g:g}" for g in gam))
    put("EDDQNnGammaZero", str(sum(1 for g in gam if g == 0)))
    fi = [e[k]["config"].get("full_info") for k in sorted(e)]
    put("EDDQNnFullInfo", str(sum(1 for f in fi if f)))
    fs = [e[k]["config"].get("feature_set") for k in sorted(e)]
    put("EDDQNnProfile", str(sum(1 for f in fs if "profile" in f)))
    put("EDDQNFeatureSets", ", ".join(fs))
    d = sel.get("DQN", {})
    put("DQNGammas", ", ".join(f"{d[k]['config'].get('gamma'):g}" for k in sorted(d)))
    for mdl in MODELS:
        if mdl in sel:
            fs = [sel[mdl][k]["config"].get("feature_set") for k in sorted(sel[mdl])]
            put(mname(MKEY[mdl], "nProfile"), str(sum(1 for f in fs if "profile" in f)))


def search_space_table():
    sp = PROTOCOL["search_spaces"]
    lines = []
    for fam, keys in (("All neural models", sp["_neural_common"]), ("All agents", sp["_agent_common"]),
                      ("E-DDQN", sp["E-DDQN"]), ("E-CB", sp["E-CB"]), ("DQN", sp["DQN"]), ("CS-MLP", sp["CS-MLP"]),
                      ("LogReg", sp["LogReg"]), ("HGB", sp["HGB"]), ("RF", sp["RF"])):
        for k, v in keys.items():
            lines.append(f"{fam} & {k.replace('_', chr(92) + '_')} & {', '.join(map(str, v))} \\\\")
    fx = PROTOCOL["fixed_training"]["agents"]
    lines.append("All agents & fixed & " + ", ".join(f"{k.replace('_', chr(92) + '_')}={v}" for k, v in fx.items()) + " \\\\")
    (SGEN / "tab_search_space.tex").write_text(
        "\\begin{longtable}{p{0.16\\textwidth}p{0.2\\textwidth}p{0.55\\textwidth}}\n\\toprule\nFamily & Hyper-parameter & Candidate values \\\\\n\\midrule\n\\endhead\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n")
    put("HPOBudget", str(PROTOCOL["hpo_budget"]["E-DDQN"]))


def ablation_tables():
    p = TAB / "ablation_ratio_summary.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    main = pd.read_csv(TAB / "metrics_summary.csv")
    full = main[(main.model == "E-DDQN")]
    rows = []
    for rule in ("D1_min_cost", "D0_bayes_1_6"):
        f = full[full.rule == rule].iloc[0]
        lines = [f"Full E-DDQN (selected) & {fmt('recall', f.recall_mean)} & {fmt('specificity', f.specificity_mean)} & "
                 f"{fmt('expected_cost', f.expected_cost_mean)}$\\pm${fmt('expected_cost', f.expected_cost_std)} & "
                 f"{fmt('roc_auc', f.roc_auc_mean)} & {fmt('pr_auc', f.pr_auc_mean)} \\\\", "\\midrule"]
        disp = {"asymmetric": "symmetric $\\pm$1 reward instead of asymmetric", "dueling": "no dueling head",
                "double": "no double estimator (max-operator target)", "soft target": "hard target copy every 10 updates",
                "class-balanced": "uniform instead of class-balanced replay", "warm-up": "no warm-up",
                "full-information": "full-information replay toggled", "gamma = 0": "$\\gamma=0$ (contextual bandit)"}
        for _, r in t[(t.rule == rule) & t.model.str.startswith("ABL")].iterrows():
            lab = r.model[4:].replace("_", " ")
            name = next((v for k, v in disp.items() if k in lab), lab)
            lines.append(f"{name} & {fmt('recall', r.recall_mean)} & {fmt('specificity', r.specificity_mean)} & "
                         f"{fmt('expected_cost', r.expected_cost_mean)}$\\pm${fmt('expected_cost', r.expected_cost_std)} & "
                         f"{fmt('roc_auc', r.roc_auc_mean)} & {fmt('pr_auc', r.pr_auc_mean)} \\\\")
        tag = "D1" if rule.startswith("D1") else "D0"
        short = {"asymmetric": "Asym", "dueling": "Duel", "double": "Double", "soft target": "Soft",
                 "class-balanced": "Bal", "warm-up": "Warm", "full-information": "Full", "gamma = 0": "GammaZero"}
        for _, r in t[(t.rule == rule) & t.model.str.startswith("ABL")].iterrows():
            lab = r.model[4:].replace("_", " ")
            key = next((v for k, v in short.items() if k in lab), None)
            if key:
                for m in ("expected_cost", "roc_auc", "recall", "specificity", "pr_auc"):
                    put(mname("Abl", key, RULES[rule], METRICS[m]), fmt(m, r[f"{m}_mean"]))
        (GEN / f"tab_ablation_{tag}.tex").write_text(
            "\\begin{tabular}{lccccc}\n\\toprule\nVariant & Recall & Spec. & Cost$_{5:1}$ & ROC-AUC & PR-AUC \\\\\n\\midrule\n" +
            "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    lines = []
    for rule in ("D0_bayes_1_6", "D1_min_cost"):
        for _, r in t[(t.rule == rule) & t.model.str.startswith("RATIO")].sort_values("model", key=lambda s: s.str[6:].astype(float)).iterrows():
            ratio = r.model.split()[1]
            lines.append(f"{ratio}:1 & {'untuned (argmax $Q$)' if rule.startswith('D0') else 'validation min-cost'} & "
                         f"{fmt('recall', r.recall_mean)} & {fmt('specificity', r.specificity_mean)} & "
                         f"{fmt('expected_cost', r.expected_cost_mean)} & {fmt('roc_auc', r.roc_auc_mean)} \\\\")
            put(mname("Ratio", ratio, RULES[rule], "Recall"), fmt("recall", r.recall_mean))
            put(mname("Ratio", ratio, RULES[rule], "Spec"), fmt("specificity", r.specificity_mean))
            put(mname("Ratio", ratio, RULES[rule], "Cost"), fmt("expected_cost", r.expected_cost_mean))
        lines.append("\\midrule")
    (GEN / "tab_ratio.tex").write_text(
        "\\begin{tabular}{llcccc}\n\\toprule\nReward FN:FP regret & Operating rule & Recall & Spec. & Cost$_{5:1}$ & ROC-AUC \\\\\n\\midrule\n" +
        "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def attack_tables():
    p = TAB / "attack_summary.csv"
    if not p.exists():
        return
    a = pd.read_csv(p)
    w = pd.read_csv(TAB / "attack_worst_case.csv").set_index("model")
    show = [("fps_cap", 45.0), ("rot_jitter", 0.1), ("speed_scale_up", 0.5), ("accel_spikes", 40.0)]
    lines = []
    for mdl in MODELS:
        g = a[(a.model == mdl) & (a.window == "whole")]
        if g.empty:
            continue
        cells = []
        for kind, val in show:
            r = g[(g.kind == kind) & (np.isclose(g.value, val))].iloc[0]
            cells.append(f"{100 * r.recall_attacked:.1f}/{100 * r.fpr_attacked:.1f}")
        r0 = g.iloc[0]
        ww = w.loc[mdl]
        lines.append(f"{mdl} & {100 * r0.recall_clean:.1f}/{100 * r0.fpr_clean:.1f} & " + " & ".join(cells) +
                     f" & {100 * ww.recall_worst_case_evasion:.1f} & {100 * ww.fpr_worst_case_false_alarm:.1f} \\\\")
        k = MKEY[mdl]
        put(mname(k, "AttWorstRecall"), f"{100 * ww.recall_worst_case_evasion:.1f}")
        put(mname(k, "AttWorstFPR"), f"{100 * ww.fpr_worst_case_false_alarm:.1f}")
        put(mname(k, "AttCleanRecall"), f"{100 * ww.recall_clean:.1f}")
        put(mname(k, "AttCleanFPR"), f"{100 * ww.fpr_clean:.1f}")
        for kind, val in show:
            r = g[(g.kind == kind) & (np.isclose(g.value, val))].iloc[0]
            put(mname(k, "Att", kind, "Recall"), f"{100 * r.recall_attacked:.1f}")
            put(mname(k, "Att", kind, "FPR"), f"{100 * r.fpr_attacked:.1f}")
    drops = 100 * (w.recall_clean - w.recall_worst_case_evasion)
    put("AttEvasionDropMin", f"{drops.min():.0f}")
    put("AttEvasionDropMax", f"{drops.max():.0f}")
    (GEN / "tab_attack.tex").write_text(
        "\\begin{tabular}{lcccccccc}\n\\toprule\n & Clean & fps$\\le$45 & jitter 0.10 & speed $\\times$1.5 & spikes 40 & \\multicolumn{2}{c}{Adaptive worst case} \\\\\n"
        "Model & Rec./FPR & Rec./FPR & Rec./FPR & Rec./FPR & Rec./FPR & Rec. (evasion) & FPR (false alarm) \\\\\n\\midrule\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    full = a.copy()
    full["Recall clean"] = (100 * full.recall_clean).round(1)
    full["Recall attacked"] = (100 * full.recall_attacked).round(1)
    full["FPR clean"] = (100 * full.fpr_clean).round(1)
    full["FPR attacked"] = (100 * full.fpr_attacked).round(1)
    lines = []
    for _, r in full.sort_values(["model", "kind", "value", "window"]).iterrows():
        lines.append(f"{r.model} & {r.kind.replace('_', ' ')} & {r.value:g} & {r.window} & {r['Recall clean']} & {r['Recall attacked']} & "
                     f"{r['FPR clean']} & {r['FPR attacked']} & {r.cost_clean:.3f} & {r.cost_attacked:.3f} \\\\")
    (SGEN / "tab_attack_full.tex").write_text(
        "\\begin{longtable}{llrlrrrrrr}\n\\toprule\nModel & Scenario & Value & Window & Rec. clean & Rec. att. & FPR clean & FPR att. & Cost clean & Cost att. \\\\\n\\midrule\n\\endhead\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n")


def diag_macros():
    p = TAB / "overestimation.csv"
    if p.exists():
        o = pd.read_csv(p)
        lines = []
        for mdl, g in o.groupby("model"):
            lines.append(f"{mdl} & {g.gamma.min():g}--{g.gamma.max():g} & {g.mean_Q_greedy.mean():.3f} & {g.mean_MC_return.mean():.3f} & "
                         f"{g.overestimation.mean():+.3f} $\\pm$ {g.overestimation.std():.3f} \\\\")
            put(mname(MKEY[mdl], "Overest"), f"{g.overestimation.mean():+.3f}")
        (SGEN / "tab_overestimation.tex").write_text(
            "\\begin{tabular}{lcccc}\n\\toprule\nModel & $\\gamma$ (selected) & mean $Q(s,\\pi(s))$ & mean MC return & bias (mean $\\pm$ SD over runs) \\\\\n\\midrule\n" +
            "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    p = TAB / "training_diagnostics.csv"
    if p.exists():
        d = pd.read_csv(p)
        for mdl, g in d.groupby("model"):
            if mdl in MKEY:
                put(mname(MKEY[mdl], "MedBestEpoch"), f"{g.best_epoch.median():.0f}" if g.best_epoch.notna().any() else "--")
                put(mname(MKEY[mdl], "MeanSeconds"), f"{g.seconds.mean():.0f}")
                if "q_abs_max" in g:
                    put(mname(MKEY[mdl], "QAbsMax"), f"{g.q_abs_max.max():.2f}")
    p = TAB / "shap_margin_summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        top = s.head(5)
        tex = lambda f: display_name(f).replace("_", chr(92) + "_")
        put("ShapTopFive", ", ".join(f"\\emph{{{tex(f)}}}" for f in top.feature))
        put("ShapTopOne", f"\\emph{{{tex(top.feature.iloc[0])}}}")


def per_fold_table():
    t = pd.read_csv(TAB / "metrics_per_outer_fold_D1.csv")
    lines = []
    for mdl in MODELS:
        g = t[t.model == mdl].sort_values("outer")
        if g.empty:
            continue
        lines.append(f"{mdl} & " + " & ".join(f"{fmt('expected_cost', c)} / {fmt('roc_auc', a)}" for c, a in zip(g.expected_cost, g.roc_auc)) + " \\\\")
    (SGEN / "tab_per_fold.tex").write_text(
        "\\begin{tabular}{l" + "c" * PROTOCOL["k_outer"] + "}\n\\toprule\nModel & " +
        " & ".join(f"Fold {k}" for k in range(PROTOCOL["k_outer"])) + " \\\\\n\\midrule\n" + "\n".join(lines) +
        "\n\\bottomrule\n\\end{tabular}\n")


def attack_auc_table():
    p = TAB / "attack_threshold_free.csv"
    if not p.exists():
        return
    a = pd.read_csv(p)
    show = [("clean", None, "Clean"), ("fps_cap", 45.0, "fps$\\le$45"), ("rot_jitter", 0.1, "jitter 0.10"),
            ("speed_scale_up", 0.5, "speed $\\times$1.5"), ("accel_spikes", 40.0, "spikes 40"),
            ("speed_scale_down", -0.5, "speed $\\times$0.5"), ("rot_smooth", 0.25, "smooth 0.25")]
    lines = []
    for mdl in MODELS:
        g = a[a.model == mdl]
        if g.empty:
            continue
        cells = []
        for kind, val, _ in show:
            r = g[g.kind == kind] if val is None else g[(g.kind == kind) & np.isclose(g.value, val)]
            cells.append(f"{r.roc_auc.iloc[0]:.3f}")
            put(mname(MKEY[mdl], "AttAuc", kind), f"{r.roc_auc.iloc[0]:.3f}")
        d = g[g.kind != "clean"].roc_auc - g[g.kind == "clean"].roc_auc.iloc[0]
        put(mname(MKEY[mdl], "AttAucMaxDrop"), f"{-d.min():.3f}")
        put(mname(MKEY[mdl], "AttAucMaxRise"), f"{d.max():.3f}")
        lines.append(f"{mdl} & " + " & ".join(cells) + f" & {d.min():+.3f} / {d.max():+.3f} \\\\")
    allchg = []
    for mdl in MODELS:
        g = a[a.model == mdl]
        if not g.empty:
            allchg += list(g[g.kind != "clean"].roc_auc - g[g.kind == "clean"].roc_auc.iloc[0])
    put("AttAucMaxAbsChange", f"{max(abs(x) for x in allchg):.3f}")
    put("AttAucFracSmall", f"{100 * np.mean([abs(x) < 0.03 for x in allchg]):.0f}")
    put("AttAucNVariants", str(a[a.kind != "clean"].variant.nunique()))   # whole-session variants in the min/max column
    head = " & ".join(h for _, _, h in show)
    (GEN / "tab_attack_auc.tex").write_text(
        "\\begin{tabular}{l" + "c" * (len(show) + 1) + "}\n\\toprule\nModel & " + head + " & min / max $\\Delta$ \\\\\n\\midrule\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def optimism_table():
    p = TAB / "optimism_gap.csv"
    if not p.exists():
        return
    o = pd.read_csv(p).groupby("model")[["inner_oof_cost_at_selection", "outer_test_cost", "optimism_gap"]].mean()
    lines = []
    for mdl in MODELS:
        if mdl not in o.index:
            continue
        r = o.loc[mdl]
        lines.append(f"{mdl} & {r.inner_oof_cost_at_selection:.3f} & {r.outer_test_cost:.3f} & {r.optimism_gap:+.3f} \\\\")
        put(mname(MKEY[mdl], "OptGap"), f"{r.optimism_gap:+.3f}")
        put(mname(MKEY[mdl], "OptInner"), f"{r.inner_oof_cost_at_selection:.3f}")
    (GEN / "tab_optimism.tex").write_text(
        "\\begin{tabular}{lccc}\n\\toprule\nModel & Inner-OOF cost at selection & Outer-test cost & Gap \\\\\n\\midrule\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def ratio_argmax_table():
    p = TAB / "ratio_argmax_policy.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    summ = pd.read_csv(TAB / "ablation_ratio_summary.csv")
    lines = []
    for _, r in d.iterrows():
        R = int(r.ratio)
        tuned = summ[(summ.model == f"RATIO {R}") & (summ.rule == "D1_min_cost")].iloc[0]
        k = mname("RatioArg", str(R))
        put(k + "Recall", f"{100 * r.recall_mean:.1f}")
        put(k + "Spec", f"{100 * r.specificity_mean:.1f}")
        put(k + "Cost", f"{r.expected_cost_mean:.3f}")
        put(mname("RatioTuned", str(R), "Cost"), f"{tuned.expected_cost_mean:.3f}")
        put(mname("RatioTuned", str(R), "RocAuc"), f"{tuned.roc_auc_mean:.3f}")
        lines.append(f"{R}:1 & $1/{R + 1}$ & {100 * r.recall_mean:.1f}$\\pm${100 * r.recall_std:.1f} & {100 * r.specificity_mean:.1f}$\\pm${100 * r.specificity_std:.1f} & "
                     f"{r.expected_cost_mean:.3f} & {100 * tuned.recall_mean:.1f} & {100 * tuned.specificity_mean:.1f} & {tuned.expected_cost_mean:.3f}$\\pm${tuned.expected_cost_std:.3f} & {tuned.roc_auc_mean:.3f} \\\\")
    (GEN / "tab_ratio_policy.tex").write_text(
        "\\begin{tabular}{lcccccccc}\n\\toprule\n & & \\multicolumn{3}{c}{Greedy policy ($\\arg\\max_a Q$)} & \\multicolumn{3}{c}{Validation-tuned threshold (D1)} & \\\\\n"
        "FN:FP regret & implied cut-off & Recall & Spec. & Cost$_{5:1}$ & Recall & Spec. & Cost$_{5:1}$ & ROC-AUC \\\\\n\\midrule\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def full_metric_table():
    """Supplement: every requested metric, rule D1 (mean +- SD over seeds) and the
    exact confusion-matrix counts of the seed ensemble."""
    summ = pd.read_csv(TAB / "metrics_summary.csv")
    s = summ[summ.rule == "D1_min_cost"].set_index("model")
    cols = [("accuracy", "Acc."), ("precision", "Prec."), ("recall", "Rec."), ("specificity", "Spec."), ("f1", "F1"),
            ("balanced_accuracy", "Bal.acc."), ("mcc", "MCC"), ("g_mean", "G-mean"), ("fpr", "FPR"), ("fnr", "FNR"),
            ("expected_cost", "Cost$_{5:1}$")]
    lines = []
    for mdl in MODELS:
        if mdl not in s.index:
            continue
        r = s.loc[mdl]
        cells = [f"{fmt(m, r[f'{m}_mean'])}$\\pm${fmt(m, r[f'{m}_std'], sd=True)}" for m, _ in cols]
        cm = [fmt(m, r[f"ens_{m}"]) for m in ("TP", "TN", "FP", "FN")]
        lines.append(f"{mdl} & " + " & ".join(cells) + " & " + " & ".join(cm) + " \\\\")
    head = " & ".join(h for _, h in cols)
    (SGEN / "tab_full_metrics.tex").write_text(
        "\\begin{tabular}{l" + "c" * (len(cols) + 4) + "}\n\\toprule\nModel & " + head + " & TP & TN & FP & FN \\\\\n\\midrule\n" +
        "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    data_macros()
    leakage_macros_and_table()
    if (TAB / "metrics_summary.csv").exists():
        main_macros_and_tables()
        ci_macros()
        paired_tables()
        matched_tables()
        event_table()
        selected_config_table()
        search_space_table()
        ablation_tables()
        attack_tables()
        diag_macros()
        per_fold_table()
        attack_auc_table()
        optimism_table()
        ratio_argmax_table()
        full_metric_table()
    # typographic minus for negative numbers (macros and generated tables)
    neg = re.compile(r"(?<![\w$\-{])-(?=\d)")
    for k in list(macros):
        macros[k] = neg.sub("$-$", macros[k])
    for d in (GEN, SGEN):
        for f in d.glob("tab_*.tex"):
            f.write_text(neg.sub("$-$", f.read_text()))
    out = ["% AUTO-GENERATED by scripts/05_make_tables.py -- do not edit"]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (GEN / "numbers.tex").write_text("\n".join(out) + "\n")
    (SGEN / "numbers.tex").write_text("\n".join(out) + "\n")
    (RESULTS / "tables" / "manuscript_macros.json").write_text(json.dumps(macros, indent=1))
    print(f"{len(macros)} macros written")
