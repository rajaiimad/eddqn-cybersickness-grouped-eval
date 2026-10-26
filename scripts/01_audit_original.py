"""Phase 1 - audit of the original repository, data provenance and reported numbers.

Outputs (results/audit/):
  inventory.csv                 every tracked file with size and SHA-256
  provenance.json               curated release <-> raw exports <-> dataset.csv
  manuscript_number_check.csv   recomputation of every confusion-matrix based number
  leakage_demo.csv              the original 'oversample -> random split' protocol
                                reconstructed vs. a grouped split (same model)
  leakage_demo_duplicates.json  exact train/test duplicates under that protocol
  proxy_original_splits/        X_/y_{train,val,test}.csv reconstructed for
                                running the unmodified original script
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn import RAW_ROOT, REPO_ROOT, RESULTS  # noqa: E402
from eddqn.data import build_datasets, grouped_folds  # noqa: E402
from eddqn.io_raw import load_raw_sessions, link_selected_to_raw, verify_dataset_csv  # noqa: E402
from eddqn.metrics import binary_metrics, mcnemar, ranking_metrics  # noqa: E402

OUT = RESULTS / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def inventory():
    r = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True)
    files = r.stdout.split("\n") if r.returncode == 0 and r.stdout.strip() else \
        sorted(str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob("*")      # archive without git history
               if p.is_file() and ".git" not in p.parts)
    rows = []
    for f in files:
        if not f or not (f == "DRL_cybersickness_FAST_v2.py" or f.startswith("UFFCSData-master_raw_source/")):
            continue
        p = REPO_ROOT / f
        rows.append({"path": f, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inventory.csv", index=False)
    top = df.assign(top=df.path.str.split("/").str[:2].str.join("/")).groupby("top").agg(files=("path", "size"), bytes=("bytes", "sum"))
    return df, top


def provenance():
    raw = load_raw_sessions()
    link = link_selected_to_raw(raw)
    cur = raw.merge(link[["session_id", "selected_id"]], on="session_id")
    chk = verify_dataset_csv(cur)
    grades = cur["DiscomfortLevel"].value_counts().sort_index().to_dict()
    n0 = grades[0]
    recon = {
        "curated_rows": int(len(cur)), "curated_sessions": int(cur.session_id.nunique()),
        "raw_sessions_total": int(raw.session_id.nunique()), "raw_rows_total": int(len(raw)),
        "curated_grade_counts": {int(k): int(v) for k, v in grades.items()},
        "manuscript_total_samples": 20972, "four_times_grade0": 4 * n0,
        "manuscript_train_per_grade": 3670, "0.7_times_grade0": 0.7 * n0,
        "oversampling_factor_needed": {int(g): round(n0 / c, 3) for g, c in grades.items()},
        "binary_prevalence_raw": float((cur.DiscomfortLevel >= 1).mean()),
        "binary_prevalence_manuscript_test": 2359 / 3146,
        "dataset_csv_check": chk,
        "selected_to_raw": link.to_dict("records"),
    }
    (OUT / "provenance.json").write_text(json.dumps(recon, indent=2, default=str))
    return recon


def manuscript_numbers():
    """Recompute the numbers printed in the submitted manuscript from its own
    confusion matrices (Tables 6-9, Sec. V-E)."""
    def m(tp, tn, fp, fn):
        y = np.r_[np.ones(tp + fn), np.zeros(tn + fp)].astype(int)
        p = np.r_[np.ones(tp), np.zeros(fn), np.zeros(tn), np.ones(fp)].astype(int)
        return binary_metrics(p, y)
    rows = []
    claims = {
        "E-DDQN test (Tab.8/9)": ((2348, 700, 87, 11), dict(accuracy=96.88, precision=96.43, recall=99.53, specificity=88.95, f1=97.96, mcc=0.916, balanced_accuracy=94.24, g_mean=94.09, npv=98.45, f2=98.90)),
        "Std DQN binary test (Tab.8/9)": ((2231, 743, 44, 128), dict(accuracy=94.53, precision=98.07, recall=94.57, specificity=94.41, f1=96.29, mcc=0.861, balanced_accuracy=94.49, g_mean=94.49, npv=85.30, f2=95.25)),
    }
    for name, (cm, claim) in claims.items():
        got = m(*cm)
        for k, v in claim.items():
            val = got[k] if k == "mcc" else 100 * got[k]
            rows.append({"item": name, "metric": k, "manuscript": v, "recomputed": round(val, 3),
                         "consistent": abs(val - v) < (0.0015 if k == "mcc" else 0.015)})
    # McNemar as printed: b=96, c=34, chi2=28.6, p=8.8e-8
    err_base, err_e = 44 + 128, 87 + 11
    b, c = 96, 34
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    rows.append({"item": "McNemar (Sec. V-E)", "metric": "chi2 from printed b,c", "manuscript": 28.6,
                 "recomputed": round(chi2, 3), "consistent": abs(chi2 - 28.6) < 0.05})
    rows.append({"item": "McNemar (Sec. V-E)", "metric": "b - c implied by confusion matrices",
                 "manuscript": b - c, "recomputed": err_base - err_e, "consistent": (b - c) == (err_base - err_e)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "manuscript_number_check.csv", index=False)
    return df


def _metrics_row(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    s = model.predict_proba(Xte)[:, 1]
    r = binary_metrics((s >= 0.5).astype(int), yte)
    r.update(ranking_metrics(s, yte))
    r["protocol"] = name
    return r


def _write_proxy(d, cols, tr, va, te):
    from sklearn.preprocessing import RobustScaler
    d.mkdir(parents=True, exist_ok=True)
    sc = RobustScaler().fit(tr[cols].to_numpy(float))
    for name, part in (("train", tr), ("val", va), ("test", te)):
        pd.DataFrame(sc.transform(part[cols].to_numpy(float)), columns=cols).to_csv(d / f"X_{name}.csv", index=False)
        part[["grade"]].rename(columns={"grade": "DiscomfortLevel"}).to_csv(d / f"y_{name}.csv", index=False)


def leakage_demo():
    """Reconstruct the original protocol from dataset.csv and compare it with a
    grouped split using the SAME model and features (the 21 dataset.csv
    columns).  HistGradientBoosting is used because it needs no tuning to
    expose the effect; the point is the protocol, not the model."""
    ev, _, _ = build_datasets()
    cols = ["UserGenere", "UserAge", "UserExperience", "UserFlicker", "UserGlassesUse",
            "UserVisionProblems", "UserEyeDominance", "StaticFrame", "CameraAutoMovement",
            "RegionOfInterest", "TimeStamp", "CameraFieldOfView", "PlayerSpeed", "PlayerAcceleration",
            "CameraRotationX", "CameraRotationY", "CameraRotationZ", "PlayerPositionX",
            "PlayerPositionY", "PlayerPositionZ", "GameFps"]
    df = ev.copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes
    rng = np.random.RandomState(42)
    rows, dup = [], {}

    # (A) original protocol: random oversampling of every grade to the size of
    # grade 0 (5,243) BEFORE splitting, then stratified 70/15/15 row split.
    n0 = int((df.grade == 0).sum())
    parts = [df[df.grade == g].sample(n0, replace=(g != 0), random_state=rng) for g in range(4)]
    over = pd.concat(parts).reset_index(drop=True)
    tr, tmp = train_test_split(over, test_size=0.30, stratify=over.grade, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.grade, random_state=42)
    dup["oversampled_total"] = len(over)
    dup["split_sizes"] = [len(tr), len(va), len(te)]
    dup["test_rows_with_exact_copy_in_train"] = int(te.merge(tr[cols].drop_duplicates(), on=cols).shape[0])
    dup["test_positive_rows_with_exact_copy_in_train"] = int(te[te.y == 1].merge(tr[cols].drop_duplicates(), on=cols).shape[0])
    dup["test_positive_rows"] = int((te.y == 1).sum())
    dup["test_sessions_also_in_train"] = int(len(set(te.session_id) & set(tr.session_id)))
    hgb = lambda: HistGradientBoostingClassifier(random_state=0)
    rows.append(_metrics_row("A: oversample-then-row-split (original)", hgb(), tr[cols], tr.y, te[cols], te.y))
    # proxy files for the unmodified original script (robust scaling fitted on
    # the training split, as described in the manuscript)
    _write_proxy(OUT / "proxy_original_splits" / "leaky_oversampled", cols, tr, va, te)

    # (B) no oversampling, random row-level stratified split (temporal leakage only)
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.grade, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.grade, random_state=42)
    rows.append(_metrics_row("B: row-level random split, no oversampling", hgb(), tr[cols], tr.y, te[cols], te.y))

    # (C) grouped 5-fold (linkage groups), same columns -> pooled OOF
    folds = grouped_folds(df, 5, 20261012)
    pr, ys = [], []
    for k in range(5):
        teg = folds.loc[folds.fold == k, "group_id"]
        te, tr = df[df.group_id.isin(teg)], df[~df.group_id.isin(teg)]
        m = hgb().fit(tr[cols], tr.y)
        pr.append(m.predict_proba(te[cols])[:, 1])
        ys.append(te.y.to_numpy())
    s, y = np.concatenate(pr), np.concatenate(ys)
    r = binary_metrics((s >= 0.5).astype(int), y)
    r.update(ranking_metrics(s, y))
    r["protocol"] = "C: grouped 5-fold (linkage groups), same columns"
    rows.append(r)
    # grouped proxy split for the original script: test = fold 0, val = fold 1, train = folds 2-4
    g = lambda ks: df[df.group_id.isin(folds.loc[folds.fold.isin(ks), "group_id"])]
    _write_proxy(OUT / "proxy_original_splits" / "grouped", cols, g([2, 3, 4]), g([1]), g([0]))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "leakage_demo.csv", index=False)
    (OUT / "leakage_demo_duplicates.json").write_text(json.dumps(dup, indent=2))
    return out, dup


if __name__ == "__main__":
    inv, top = inventory()
    print(top)
    print(json.dumps({k: v for k, v in provenance().items() if k != "selected_to_raw"}, indent=1, default=str)[:2500])
    print(manuscript_numbers().to_string())
    demo, dup = leakage_demo()
    print(demo[["protocol", "accuracy", "recall", "specificity", "f1", "mcc", "roc_auc", "pr_auc"]].round(4).to_string())
    print(dup)
