"""Independent verification of the evaluation boundaries and of the replay
buffer on the REAL folds of the evaluation set (review round 4, problems 2-3).

It re-uses the production code (no re-implementation) and checks:

A. Splits      every linkage group / session / questionnaire profile lies in
               exactly one partition, for the outer split AND for all 15
               (outer, inner) splits; the saved split files match.
B. Duplicates  no raw telemetry row or derived feature vector of an
               evaluation partition is identical to a training row; distance
               from each evaluation row to its nearest training row compared
               with the distance between consecutive seconds of a session.
C. Fitting     preprocessing parameters and the training matrix are unchanged
               when every evaluation row is replaced by noise (train-only fit).
D. Selection   HPO runs hold no test scores; the selected configurations are
               re-derived from the pooled inner-validation metrics; the D1
               thresholds are re-derived from the saved validation scores and
               do not depend on test scores; checkpoints are the best
               validation epoch and the training logs hold no test metric.
E. Replay      E-DDQN, E-CB and DQN are trained (shortened to 2 epochs) with
               their selected configurations on every outer fold; every
               transition written to the replay buffer is logged and checked:
               reward = R(stored action, label of s), s/s' are rows of the
               same session one second apart, terminal flags at the last
               second of each session only, s' never from another session,
               buffer contents equal the logged rows.

Run: python scripts/15_pipeline_verification.py      (exits non-zero on failure)
Out: results/qc/pipeline_verification.json, results/qc/pipeline_verification.md
"""

from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn import CONFIGS, PROCESSED, RESULTS  # noqa: E402
from eddqn.agents import AgentConfig, ValueAgent  # noqa: E402
from eddqn.data import build_datasets, fold_frames, prepare_fold, split_tables  # noqa: E402
from eddqn.experiment import build_cache, operating_thresholds  # noqa: E402
from eddqn.features import feature_columns  # noqa: E402
from eddqn.io_raw import PROFILE  # noqa: E402

PROTOCOL = json.loads((CONFIGS / "protocol.json").read_text())
KO, KI = PROTOCOL["k_outer"], PROTOCOL["k_inner"]
SEEDS = PROTOCOL["final_seeds"]
TELEMETRY = ["TimeStamp", "CameraFieldOfView", "PlayerSpeed", "PlayerAcceleration", "CameraRotationX",
             "CameraRotationY", "CameraRotationZ", "RegionOfInterest", "GameFps",
             "PlayerPositionX", "PlayerPositionY", "PlayerPositionZ"]
TELEMETRY_NO_TIME = [c for c in TELEMETRY if c != "TimeStamp"]
RUNS = RESULTS / "runs"
OUT = RESULTS / "qc"
checks, info = [], {}


def check(name, cond, detail=""):
    checks.append({"check": name, "pass": bool(cond), "detail": str(detail)})
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def row_keys(df, cols):
    return set(map(tuple, df[cols].round(9).to_numpy().tolist()))


# ------------------------------------------------------------------ A
def verify_splits(long, tables):
    outer = tables["outer"]
    check("A1 every linkage group is in exactly one outer fold",
          outer.group_id.is_unique and outer.fold.notna().all() and set(outer.group_id) == set(long.group_id),
          f"{len(outer)} groups")
    check("A2 every session belongs to exactly one linkage group",
          (long.groupby("session_id").group_id.nunique() == 1).all(), f"{long.session_id.nunique()} sessions")
    saved_o = pd.read_csv(PROCESSED / "outer_folds.csv")
    saved_i = pd.read_csv(PROCESSED / "inner_folds.csv")
    check("A3 saved outer split file equals the recomputed split",
          saved_o.set_index("group_id").fold.sort_index().equals(outer.set_index("group_id").fold.sort_index()))
    ok_inner = all(
        saved_i[saved_i.outer_fold == k].set_index("group_id").fold.sort_index()
        .equals(tables["inner"][k].set_index("group_id").fold.sort_index()) for k in range(KO))
    check("A4 saved inner split file equals the recomputed split", ok_inner)
    prof = long.assign(profile=list(map(tuple, long[PROFILE].astype(str).to_numpy())))
    rows = []
    for k in range(KO):
        dev, test = fold_frames(prof, tables, k)
        inner = tables["inner"][k]
        check(f"A5 outer {k}: inner folds partition exactly the development groups",
              set(inner.group_id) == set(dev.group_id) and inner.group_id.is_unique)
        for j in range(KI):
            tr, val, te = fold_frames(prof, tables, k, j)
            parts = {"train": tr, "val": val, "test": te}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                A, B = parts[a], parts[b]
                rows.append({"outer": k, "inner": j, "pair": f"{a}/{b}",
                             "sessions": len(set(A.session_id) & set(B.session_id)),
                             "groups": len(set(A.group_id) & set(B.group_id)),
                             "profiles": len(set(A.profile) & set(B.profile))})
            check(f"A6 outer {k} inner {j}: partitions cover the data once",
                  len(tr) + len(val) + len(te) == len(long)
                  and not (set(tr.index) & set(val.index)) and not (set(tr.index) & set(te.index)))
    ov = pd.DataFrame(rows)
    check("A7 no session, linkage group or exact questionnaire profile is shared by any two partitions "
          "(15 outer x inner splits, train/val/test)", int(ov[["sessions", "groups", "profiles"]].to_numpy().sum()) == 0,
          f"{len(ov)} partition pairs")
    info["split_pairs_checked"] = len(ov)


# ------------------------------------------------------------------ B, C
def verify_rows_and_fitting(long, tables):
    dup_rows, nn_rows = [], []
    rng = np.random.default_rng(0)
    for k in range(KO):
        dev, test = fold_frames(long, tables, k)
        for name, cols in (("telemetry incl. time", TELEMETRY), ("telemetry excl. time", TELEMETRY_NO_TIME)):
            dup_rows.append({"outer": k, "level": f"raw {name}", "boundary": "dev/test",
                             "identical_rows": len(row_keys(dev, cols) & row_keys(test, cols))})
        fd = prepare_fold(dev, {"test": test}, with_profile=True)
        Xtr, Xte = fd.X_tr, fd.X_ev["test"]
        dup_rows.append({"outer": k, "level": "derived feature vector (all features)", "boundary": "dev/test",
                         "identical_rows": len(set(map(bytes, Xtr)) & set(map(bytes, Xte)))})
        # nearest-neighbour (NN) distances in the scaled telemetry feature space
        cols = [fd.feature_names.index(c) for c in feature_columns(fd.feature_names, "telemetry")]
        A, B = Xtr[:, cols], Xte[:, cols]
        g_tr, s_tr = fd.meta_tr.group_id.to_numpy(), fd.meta_tr.session_id.to_numpy()
        d_test = NearestNeighbors(n_neighbors=1).fit(A).kneighbors(B)[0][:, 0]
        same = s_tr[1:] == s_tr[:-1]
        d_consec = np.linalg.norm(A[1:] - A[:-1], axis=1)[same]
        # reference 1: a development row to the nearest row of ANOTHER linkage group
        d_other = np.empty(len(A))
        for g in np.unique(g_tr):
            m = g_tr == g
            d_other[m] = NearestNeighbors(n_neighbors=1).fit(A[~m]).kneighbors(A[m])[0][:, 0]
        # reference 2: a development row to the nearest OTHER row of its own session
        # (what a row-level split exposes)
        d_own = np.empty(len(A))
        for sid in np.unique(s_tr):
            m = np.flatnonzero(s_tr == sid)
            d_own[m] = NearestNeighbors(n_neighbors=2).fit(A[m]).kneighbors(A[m])[0][:, 1]
        nn_rows.append({"outer": k, "test_to_nearest_dev": float(np.median(d_test)),
                        "dev_to_nearest_other_group": float(np.median(d_other)),
                        "dev_to_nearest_same_session": float(np.median(d_own)),
                        "consecutive_seconds": float(np.median(d_consec))})
        for j in range(KI):
            tr, val, te = fold_frames(long, tables, k, j)
            for nm, ev in (("val", val), ("test", te)):
                dup_rows.append({"outer": k, "inner": j, "level": "raw telemetry incl. time",
                                 "boundary": f"inner-train/{nm}",
                                 "identical_rows": len(row_keys(tr, TELEMETRY) & row_keys(ev, TELEMETRY))})
            # C: train-only fitting -- replace every evaluation row by noise
            ref = prepare_fold(tr, {"val": val, "test": te}, with_profile=True)
            noisy = {}
            for nm, ev in (("val", val), ("test", te)):
                e = ev.copy()
                for c in TELEMETRY_NO_TIME:
                    e[c] = rng.normal(0, 1e3, len(e))
                noisy[nm] = e
            alt = prepare_fold(tr, noisy, with_profile=True)
            check(f"C1 outer {k} inner {j}: learned preprocessing parameters do not depend on val/test rows",
                  json.dumps(ref.params, sort_keys=True, default=str) == json.dumps(alt.params, sort_keys=True, default=str))
            check(f"C2 outer {k} inner {j}: training matrix does not depend on val/test rows",
                  np.array_equal(ref.X_tr, alt.X_tr))
    d = pd.DataFrame(dup_rows)
    check("B1 no raw telemetry row (with or without elapsed time) or derived feature vector of an "
          "evaluation partition is identical to a training row", int(d.identical_rows.sum()) == 0,
          f"{len(d)} comparisons")
    nn = pd.DataFrame(nn_rows)
    info["nearest_neighbour"] = nn.round(4).to_dict("records")
    # Descriptive only (no pass/fail): the first version of this check compared the
    # test-to-nearest-training distance with the distance between CONSECUTIVE
    # seconds, which is ill-posed (a nearest neighbour among ~7,000 rows is
    # naturally closer than one fixed neighbour); it failed in 4 of 5 folds and
    # was replaced by the two reference distances above.  See the report.
    info["nn_note"] = ("medians; test rows are compared with the nearest development row, against the same "
                       "distance for development rows to another linkage group and within their own session")


# ------------------------------------------------------------------ D
def verify_selection(long, tables):
    cache = build_cache(long, tables, PROTOCOL, include_test=False, outers=[0])
    check("D1 HPO preprocessing caches contain no outer-test rows",
          all("test" not in fd.X_ev for fd in cache.values()))
    hpo_files = list((RUNS / "hpo").rglob("*.pkl"))
    bad = [p for p in hpo_files if any(key.startswith("test") or key == "attack_scores" for key in load(p))]
    check("D2 no HPO run stored a test score", not bad, f"{len(hpo_files)} HPO runs")
    sel = json.loads((RESULTS / "selected_configs.json").read_text())
    hpo = pd.read_csv(RESULTS / "hpo_results.csv")
    check("D3 hpo_results.csv holds only inner-validation metrics",
          not [c for c in hpo.columns if "test" in c.lower()])
    mism = []
    for fam, by in sel.items():
        h = hpo[hpo.family == fam]
        for k, v in by.items():
            hk = h[h.outer == int(k)].assign(key2=lambda x: -x.pooled_oof_pr_auc)
            best = hk.sort_values(["pooled_oof_min_cost", "key2", "config_id"]).iloc[0]
            if int(best.config_id) != int(v["config_id"]):
                mism.append((fam, k))
    check("D4 every selected configuration is the best pooled inner-validation configuration",
          not mism, f"{sum(len(v) for v in sel.values())} (family, fold) selections; mismatches {mism}")
    # thresholds and checkpoints
    val = {(k, j): fold_frames(long, tables, k, j)[1] for k in range(KO) for j in range(KI)}
    thr_err, dep, ck_bad, test_keys = 0.0, [], 0, set()
    n_ck, n_thr = 0, 0
    for fam in ["E-DDQN", "E-CB", "DQN", "CS-MLP", "LogReg", "HGB", "RF"]:
        pred = pd.read_csv(RESULTS / "predictions" / f"{fam}.csv.gz")
        for k in range(KO):
            vss = []
            vy = np.concatenate([val[(k, j)]["y"].to_numpy() for j in range(KI)])
            for s in SEEDS:
                parts = [load(RUNS / "final" / fam / f"o{k}_s{s}_i{j}.pkl") for j in range(KI)]
                vs = np.concatenate([p["val_score"] for p in parts])
                vss.append(vs)
                th = operating_thresholds(vs, vy, PROTOCOL)["D1_min_cost"]
                stored = pred[(pred.outer == k) & (pred.seed.astype(str) == str(s))]["thr_D1_min_cost"].unique()
                thr_err = max(thr_err, float(np.max(np.abs(stored - th))))
                n_thr += 1
                for p in parts:
                    h = [r for r in p.get("history", []) if "val_min_cost" in r]
                    test_keys |= {key for r in p.get("history", []) for key in r if "test" in key}
                    if h:
                        n_ck += 1
                        best = min(h, key=lambda r: (r["val_min_cost"], -r["val_pr_auc"]))
                        ck_bad += int(best["epoch"] != p["best_epoch"])
            # seed ensemble: threshold on the mean inner-validation score
            th = operating_thresholds(np.mean(vss, axis=0), vy, PROTOCOL)["D1_min_cost"]
            stored = pred[(pred.outer == k) & (pred.seed.astype(str) == "ens")]["thr_D1_min_cost"].unique()
            thr_err = max(thr_err, float(np.max(np.abs(stored - th))))
            n_thr += 1
    check("D5 stored D1 thresholds (per seed and seed ensemble) are re-derived exactly from the saved "
          "inner-validation scores", thr_err < 1e-9, f"{n_thr} thresholds; max |difference| {thr_err:.2e}")
    check("D6 neural checkpoints are the best inner-validation epoch", ck_bad == 0, f"{n_ck} training runs")
    check("D7 no training log records a test metric", not test_keys)
    # thresholds use only validation scores: the function has no test argument
    import inspect
    sig = list(inspect.signature(operating_thresholds).parameters)
    check("D8 the threshold function receives only validation scores and labels",
          sig == ["oof_score", "oof_y", "protocol"], sig)


# ------------------------------------------------------------------ E
def verify_replay(long, tables):
    sel = json.loads((RESULTS / "selected_configs.json").read_text())
    summary = []
    for fam in ["E-DDQN", "E-CB", "DQN"]:
        for k in range(KO):
            tr, val, _ = fold_frames(long, tables, k, 0)
            fd = prepare_fold(tr, {"val": val}, with_profile=True)
            cfg = dict(sel[fam][str(k)]["config"])
            cols = [fd.feature_names.index(c) for c in feature_columns(fd.feature_names, cfg["feature_set"])]
            X = fd.X_tr[:, cols]
            meta = fd.meta_tr
            c = AgentConfig(**{a: (tuple(b) if a == "hidden" else b) for a, b in cfg.items()
                               if a in AgentConfig.__dataclass_fields__})
            c.epochs, c.seed, c.eval_every = 2, 1000, 1
            agent = ValueAgent(X.shape[1], c)
            log = []
            orig = agent._store

            def logged(buf, X_, si, ni, dn, a, y, _orig=orig, _log=log, _agent=agent):
                r = _orig(buf, X_, si, ni, dn, a, y)
                _log.append((si.copy(), ni.copy(), dn.copy(), a.copy(), y.copy(), r.copy()))
                if _agent.cfg.full_info:
                    _log.append((si.copy(), ni.copy(), dn.copy(), 1 - a, y.copy(), _agent._r(1 - a, y)))
                _agent._last_buf = buf
                return r

            agent._store = logged
            agent.fit(X, fd.y_tr, meta.session_id.to_numpy(), fd.X_ev["val"][:, cols], fd.y_ev["val"])
            si, ni, dn, a, y, r = (np.concatenate([e[i] for e in log]) for i in range(6))
            sess, step = meta.session_id.to_numpy(), meta.step.to_numpy()
            last = np.r_[sess[1:] != sess[:-1], True]
            rt = agent.rt
            exp_r = np.where(a == 1, np.where(y == 1, rt["tp"], rt["fp"]), np.where(y == 1, rt["fn"], rt["tn"]))
            nt = dn == 0
            buf = agent._last_buf
            n = len(si)
            res = {
                "reward matches the stored action and the label of s": np.allclose(r, exp_r),
                "stored label is the label of s": np.array_equal(y, fd.y_tr[si]),
                "non-terminal s' is the same session": np.array_equal(sess[ni[nt]], sess[si[nt]]),
                "non-terminal s' is the next second": np.array_equal(step[ni[nt]], step[si[nt]] + 1),
                "terminal flag exactly at the last second of a session": np.array_equal(dn == 1, last[si]),
                "buffer holds the logged s, a, r, s', done (no overwrite)": (
                    n <= buf.cap and buf.n == n and np.array_equal(buf.s[:n], X[si]) and np.array_equal(buf.s2[:n], X[ni])
                    and np.array_equal(buf.a[:n], a) and np.allclose(buf.r[:n], r) and np.array_equal(buf.d[:n], dn)),
            }
            for name, ok in res.items():
                check(f"E {fam} outer {k}: {name}", ok)
            # every session is a contiguous, time-ordered block in the training matrix
            check(f"E {fam} outer {k}: sessions are contiguous and time-ordered in the training rows",
                  meta.groupby("session_id").step.apply(lambda s: (np.diff(s.to_numpy()) == 1).all()).all()
                  and (meta.session_id != meta.session_id.shift()).sum() == meta.session_id.nunique())
            summary.append({"family": fam, "outer": k, "gamma": c.gamma, "full_info": c.full_info,
                            "transitions_checked": int(n), "terminal": int((dn == 1).sum()),
                            "counterfactual_share": float(0.5 if c.full_info else 0.0)})
    info["replay"] = summary


def md_table(df):
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns),
                      *["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]])


def write_report():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pipeline_verification.json").write_text(json.dumps({"checks": checks, "info": info}, indent=2))
    n_pass = sum(c["pass"] for c in checks)
    rep = pd.DataFrame(info["replay"])
    nn = pd.DataFrame(info["nearest_neighbour"])
    lines = [
        "# Pipeline verification on the real folds (review round 4, problems 2 and 3)",
        "",
        f"Generated by `scripts/15_pipeline_verification.py`: **{n_pass} of {len(checks)} checks pass.**",
        "It reuses the production code; nothing in this check trains a reported model.",
        "",
        "## Scope and limits",
        "- The checks show that no session, linkage group, exact questionnaire profile, raw telemetry row or",
        "  derived feature vector crosses a training/evaluation boundary, and that every learned quantity",
        "  (preprocessing, configuration, checkpoint, threshold) is derived from training or inner-validation data.",
        "- They cannot show that two linkage groups never contain the same person: the release has no",
        "  participant identifiers, and a person who answered the questionnaire differently in two",
        "  non-consecutive sessions would not be linked. The evaluation is session-linkage-grouped, not",
        "  verified participant-independent.",
        "",
        "## Nearest-neighbour distances (scaled telemetry features, medians per outer fold; descriptive)",
        md_table(nn),
        "",
        "`test_to_nearest_dev`: each outer-test row to its nearest development (training) row.",
        "`dev_to_nearest_other_group`: each development row to its nearest row of another linkage group,",
        "i.e. the similarity expected between unrelated groups. `dev_to_nearest_same_session`: each development",
        "row to the nearest other row of its own session, i.e. what a row-level split exposes.",
        "",
        "Reading: test rows are about as far from their nearest training row as development rows are from",
        "another linkage group (folds 0 and 2: farther; folds 1, 3 and 4: closer, and in folds 1 and 3 about",
        "as close as rows of the same session). In this feature space, rows of one session are only modestly",
        "closer than rows of different groups, so this distance discriminates weakly: it gives no sign of",
        "near-duplicate rows across the boundary, but it cannot exclude that a test group contains the same",
        "person as a training group.",
        "",
        "Note on an earlier version of this check: it required the test-to-nearest-training distance to exceed",
        "the distance between consecutive seconds of a session, and failed in 4 of 5 folds. That criterion was",
        "ill-posed (a nearest neighbour among about 7,000 rows is naturally closer than one fixed neighbour), so",
        "it was replaced by the two reference distances above and is not used as a pass/fail check.",
        "",
        "## Replay transitions checked (2-epoch training with each fold's selected configuration, inner split 0)",
        md_table(rep),
        "",
        "With full-information replay, half of the stored transitions carry the action that was not executed,",
        "each with its own reward R(a, y); no stored transition pairs an action with another action's reward.",
        "",
        "## All checks",
        "| check | result | detail |", "|---|---|---|",
        *[f"| {c['check']} | {'pass' if c['pass'] else 'FAIL'} | {c['detail']} |" for c in checks],
    ]
    (OUT / "pipeline_verification.md").write_text("\n".join(lines) + "\n")
    write_macros(n_pass, rep, nn)
    return n_pass == len(checks)


def write_macros(n_pass, rep, nn):
    """Numbers quoted in the manuscript and supplement, written as LaTeX macros."""
    d = {c["check"].split()[0]: c["detail"] for c in checks}
    m = {
        "PVChecksPass": n_pass, "PVChecks": len(checks),
        "PVSplitPairs": info["split_pairs_checked"],
        "PVHPORuns": f"{int(d['D2'].split()[0]):,}",
        "PVCheckpointRuns": d["D6"].split()[0],
        "PVTransitions": f"{int(rep.transitions_checked.sum()):,}",
        "PVReplayRuns": len(rep),
        "PVNNTestMin": f"{nn.test_to_nearest_dev.min():.2f}", "PVNNTestMax": f"{nn.test_to_nearest_dev.max():.2f}",
        "PVNNOtherMin": f"{nn.dev_to_nearest_other_group.min():.2f}",
        "PVNNOtherMax": f"{nn.dev_to_nearest_other_group.max():.2f}",
        "PVNNOwnMin": f"{nn.dev_to_nearest_same_session.min():.2f}",
        "PVNNOwnMax": f"{nn.dev_to_nearest_same_session.max():.2f}",
    }
    path = RESULTS / "tables" / "pipeline_verification" / "pv_macros.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("% generated by scripts/15_pipeline_verification.py -- do not edit\n"
                    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))


if __name__ == "__main__":
    long, _, _ = build_datasets()
    tables = split_tables(long, PROTOCOL)
    verify_splits(long, tables)
    verify_rows_and_fitting(long, tables)
    verify_selection(long, tables)
    verify_replay(long, tables)
    ok = write_report()
    print(f"{sum(c['pass'] for c in checks)}/{len(checks)} checks pass")
    sys.exit(0 if ok else 1)
