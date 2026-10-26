"""Phase 2 - leakage-free preprocessing: build, verify and document.

Writes results/preprocessing/* and the frozen split tables
(data/processed/outer_folds.csv, inner_folds.csv) that every later
script reads.  All checks are executable assertions; the script fails loudly
if any leakage test fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn import CONFIGS, PROCESSED, RESULTS  # noqa: E402
from eddqn.data import build_datasets, frame_hash, prepare_fold  # noqa: E402
from eddqn.data import fold_frames, split_tables  # noqa: E402
from eddqn.features import CausalFeaturizer, feature_columns  # noqa: E402

OUT = RESULTS / "preprocessing"
OUT.mkdir(parents=True, exist_ok=True)
PROTOCOL = json.loads((CONFIGS / "protocol.json").read_text())


def counts(df, by):
    t = df.groupby(by).agg(sessions=("session_id", "nunique"), samples=("y", "size"),
                           positives=("y", "sum"),
                           grade0=("grade", lambda g: int((g == 0).sum())), grade1=("grade", lambda g: int((g == 1).sum())),
                           grade2=("grade", lambda g: int((g == 2).sum())), grade3=("grade", lambda g: int((g == 3).sum())))
    t["prevalence"] = (t.positives / t.samples).round(4)
    return t.reset_index()


def quality(df):
    q = {}
    q["rows"] = int(len(df))
    q["missing_values"] = {c: int(v) for c, v in df.isna().sum().items() if v and c != "selected_id"}
    q["fps_zero_first_sample"] = int(((df.GameFps <= 0) & (df.TimeStamp < 1)).sum())
    q["fps_zero_elsewhere"] = int(((df.GameFps <= 0) & (df.TimeStamp >= 1)).sum())
    q["fps_below_80_after_start"] = int(((df.GameFps < 80) & (df.TimeStamp >= 1)).sum())
    dt = df.groupby("session_id").TimeStamp.diff().dropna()
    q["sampling_interval_s"] = {"mean": float(dt.mean()), "min": float(dt.min()), "max": float(dt.max()),
                                "non_increasing": int((dt <= 0).sum())}
    tele = ["PlayerSpeed", "PlayerAcceleration", "CameraRotationX", "CameraRotationY", "CameraRotationZ", "GameFps"]
    q["duplicate_telemetry_rows_within_session"] = int(df.duplicated(subset=tele + ["session_id", "TimeStamp"]).sum())
    q["duplicate_telemetry_rows_across_sessions"] = int(df.duplicated(subset=tele + ["TimeStamp"]).sum())
    q["grades_present"] = sorted(int(g) for g in df.grade.unique())
    sw = df.groupby("session_id").y.apply(lambda s: int((s.diff() < 0).sum()))
    q["sessions_with_1_to_0_label_transitions"] = int((sw > 0).sum())
    q["label_switches_per_session_mean"] = float(df.groupby("session_id").y.apply(lambda s: s.diff().abs().sum()).mean())
    q["sessions_never_positive"] = int((df.groupby("session_id").y.max() == 0).sum())
    first = df[df.y == 1].groupby("session_id").TimeStamp.min()
    q["onset_time_s"] = {"median": float(first.median()), "min": float(first.min()), "max": float(first.max())}
    const = {}
    for c in ["CameraFieldOfView", "StaticFrame", "CameraAutoMovement", "HapticFeedback", "DegreeOfControl",
              "DofSimulation", "Locomotion", "UserPosture", "UserSymptoms"]:
        const[c] = {g: sorted(map(str, v.unique())) for g, v in df.groupby("game")[c]}
    q["per_game_values_of_removed_constant_columns"] = const
    return q


def causality_test(df, n_checks=40, seed=0):
    """Features of sample t must be identical whether or not samples > t exist."""
    rng = np.random.default_rng(seed)
    fz = CausalFeaturizer(with_profile=True).fit(df)
    sids = df.session_id.unique()
    worst = 0.0
    for _ in range(n_checks):
        sid = rng.choice(sids)
        g = df[df.session_id == sid]
        t = int(rng.integers(1, len(g)))
        full = fz.transform(g).iloc[t - 1]
        trunc = fz.transform(g.iloc[:t]).iloc[t - 1]
        worst = max(worst, float(np.max(np.abs(full.to_numpy() - trunc.to_numpy()))))
    assert worst < 1e-9, f"feature uses future samples (max diff {worst})"
    return {"checks": n_checks, "max_abs_difference": worst}


def train_only_test(df, tables):
    """Learned preprocessing parameters must not change when validation/test
    rows are altered."""
    tr, val, test = fold_frames(df, tables, 0, 0)
    a = prepare_fold(tr, {"val": val, "test": test}, with_profile=True)
    tamper = test.copy()
    tamper["PlayerAcceleration"] = tamper["PlayerAcceleration"] * 50 + 999
    tamper["GameFps"] = 0.0
    b = prepare_fold(tr, {"val": val, "test": tamper}, with_profile=True)
    same_params = a.params == b.params
    same_train = np.array_equal(a.X_tr, b.X_tr)
    same_val = np.array_equal(a.X_ev["val"], b.X_ev["val"])
    assert same_params and same_train and same_val, "preprocessing depends on test rows"
    return {"params_identical": same_params, "train_matrix_identical": same_train,
            "val_matrix_identical": same_val}


def overlap_and_duplicates(df, tables):
    rows = []
    for k in range(PROTOCOL["k_outer"]):
        dev, test = fold_frames(df, tables, k)
        fd = prepare_fold(dev, {"test": test}, with_profile=True)
        cols = feature_columns(fd.feature_names, "telemetry+profile")
        idx = [fd.feature_names.index(c) for c in cols]
        tr_rows = {r.tobytes() for r in fd.X_tr[:, idx]}
        dup = sum(r.tobytes() in tr_rows for r in fd.X_ev["test"][:, idx])
        prof_tr = set(map(tuple, dev[[c for c in dev.columns if c.startswith("User")]].drop_duplicates().itertuples(index=False)))
        prof_te = set(map(tuple, test[[c for c in test.columns if c.startswith("User")]].drop_duplicates().itertuples(index=False)))
        rows.append({"outer_fold": k, "dev_sessions": dev.session_id.nunique(), "test_sessions": test.session_id.nunique(),
                     "dev_groups": dev.group_id.nunique(), "test_groups": test.group_id.nunique(),
                     "session_overlap": len(set(dev.session_id) & set(test.session_id)),
                     "group_overlap": len(set(dev.group_id) & set(test.group_id)),
                     "identical_profile_overlap": len(prof_tr & prof_te),
                     "test_feature_rows_identical_to_a_train_row": int(dup),
                     "test_samples": len(test), "test_positives": int(test.y.sum()),
                     "dev_samples": len(dev), "dev_positives": int(dev.y.sum())})
        for j in range(PROTOCOL["k_inner"]):
            tr, val, _ = fold_frames(df, tables, k, j)
            assert not (set(tr.group_id) & set(val.group_id)), "inner overlap"
            assert not (set(tr.group_id) & set(test.group_id)), "inner-train/test overlap"
    t = pd.DataFrame(rows)
    assert (t.session_overlap == 0).all() and (t.group_overlap == 0).all()
    return t


def feature_dictionary(fd_names):
    desc = {
        "is_flight": "1 if Flight game, 0 if Race (from the session folder name)",
        "elapsed_s": "seconds since session start (TimeStamp)",
        "speed": "PlayerSpeed at t", "accel": "PlayerAcceleration at t", "abs_accel": "|PlayerAcceleration| at t",
        "rot_x": "CameraRotationX (quaternion x) at t", "rot_y": "CameraRotationY (quaternion y) at t",
        "rot_z": "CameraRotationZ (quaternion z) at t",
        "rot_rate": "||q_t - q_{t-1}|| (camera orientation change per sample; 0 at t=0)",
        "fps": "GameFps at t; start-up artefact GameFps=0 at t<1 s imputed with the TRAINING median",
        "roi_background": "RegionOfInterest == background", "roi_middleground": "RegionOfInterest == middleground",
        "roi_foreground": "RegionOfInterest == foreground",
        "cum_rot": "sum of rot_rate over 0..t", "cum_acc_spikes": "number of acceleration spikes over 0..t",
        "cum_fps_drops": "number of samples with fps < 80 over 0..t", "speed_mean_sofar": "mean speed over 0..t",
    }
    for w in (5, 15):
        desc.update({
            f"speed_mean_{w}": f"mean speed over the trailing {w} samples (t-{w-1}..t)",
            f"speed_std_{w}": f"std of speed over the trailing {w} samples",
            f"speed_delta_{w}": f"speed_t - speed_(t-{w-1}) (first sample if t<{w-1})",
            f"absacc_mean_{w}": f"mean |acceleration| over the trailing {w} samples",
            f"absacc_max_{w}": f"max |acceleration| over the trailing {w} samples",
            f"acc_spike_rate_{w}": f"fraction of the trailing {w} samples with |acc| > tau_acc (tau_acc = 90th pct of |acc| on TRAINING rows)",
            f"rotrate_mean_{w}": f"mean rot_rate over the trailing {w} samples (motion-instability index)",
            f"rotrate_std_{w}": f"std of rot_rate over the trailing {w} samples",
            f"fps_mean_{w}": f"mean fps over the trailing {w} samples", f"fps_min_{w}": f"min fps over the trailing {w} samples",
            f"fps_drop_frac_{w}": f"fraction of the trailing {w} samples with fps < 80 (fixed threshold)",
        })
    rows = []
    for n in fd_names:
        d = desc.get(n)
        if d is None and n.startswith("prof_"):
            d = "pre-session questionnaire (CSPQ) indicator: " + n[5:]
        rows.append({"feature": n, "definition": d, "causal": True,
                     "in_exposure_set": n in feature_columns(fd_names, "exposure")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ev, dv, sessions = build_datasets()
    sessions.drop(columns=["profile"]).to_csv(OUT / "session_table.csv", index=False)
    log = sessions.attrs["link_log_eval"] + sessions.attrs["link_log_dev"]
    pd.DataFrame(log, columns=["session_a", "session_b", "reason"]).to_csv(OUT / "linkage_log.csv", index=False)

    allraw = pd.concat([ev.assign(role="evaluation"), dv.assign(role="development")])
    counts(allraw, ["role"]).to_csv(OUT / "counts_by_role.csv", index=False)
    counts(ev, ["game"]).to_csv(OUT / "counts_eval_by_game.csv", index=False)
    counts(ev, ["session_id", "group_id", "game"]).to_csv(OUT / "counts_eval_by_session.csv", index=False)
    q = {"evaluation": quality(ev), "development": quality(dv)}
    (OUT / "data_quality.json").write_text(json.dumps(q, indent=2))

    tables = split_tables(ev, PROTOCOL)
    tables["outer"].to_csv(PROCESSED / "outer_folds.csv", index=False)
    inner = pd.concat([t.assign(outer_fold=k) for k, t in tables["inner"].items()])
    inner.to_csv(PROCESSED / "inner_folds.csv", index=False)
    counts(ev.merge(tables["outer"][["group_id", "fold"]], on="group_id"), ["fold"]).to_csv(OUT / "counts_by_outer_fold.csv", index=False)

    checks = {
        "causality": causality_test(ev),
        "train_only_fitting": train_only_test(ev, tables),
        "oversampling_or_synthetic_samples": "none (class balance is handled only inside training minibatches / loss weights)",
        "evaluation_data_hash": frame_hash(ev[["session_id", "step", "TimeStamp", "DiscomfortLevel"]]),
    }
    ov = overlap_and_duplicates(ev, tables)
    ov.to_csv(OUT / "partition_overlap_checks.csv", index=False)
    checks["partition_overlap"] = ov.to_dict("records")
    fd = prepare_fold(*fold_frames(ev, tables, 0)[:1], {}, with_profile=True)
    feature_dictionary(fd.feature_names).to_csv(OUT / "feature_dictionary.csv", index=False)
    params = []
    for k in range(PROTOCOL["k_outer"]):
        dev, test = fold_frames(ev, tables, k)
        p = prepare_fold(dev, {}, with_profile=True).params
        params.append({"outer_fold": k, **{a: b for a, b in p.items() if not a.startswith("scaler")}})
    pd.DataFrame(params).to_csv(OUT / "learned_parameters_by_outer_fold.csv", index=False)
    (OUT / "leakage_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    print(json.dumps({k: v for k, v in checks.items() if k != "partition_overlap"}, indent=1, default=str))
    print(ov.to_string())
    print(counts(allraw, ["role"]).to_string())
