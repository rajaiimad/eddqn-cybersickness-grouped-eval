"""Analysis datasets, grouped splits and fold-wise (train-only) preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler

from . import PROCESSED
from .features import CausalFeaturizer
from .grouping import linkage_groups, session_table
from .io_raw import link_selected_to_raw, load_raw_sessions

POSITIVE_FROM_GRADE = 1  # DiscomfortLevel >= 1 -> positive (cybersickness event)


def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (evaluation set, development set, session table).

    * evaluation set: the dataset authors' curated release (35 Dec-2019
      seated sessions; the same samples as ``dataset.csv`` used by the
      original manuscript), parsed from XML.
    * development set: the 37 Oct-2019 standing-posture raw sessions, used
      ONLY to debug code and to calibrate search-space ranges / training
      budgets before the frozen protocol is run on the evaluation set.
      Models fitted on it are never evaluated on the evaluation set.
    * The 2 Dec-2019 raw sessions that the dataset authors left out of the
      curated release are not used at all.
    """
    raw = load_raw_sessions()
    link = link_selected_to_raw(raw)
    raw = raw.merge(link[["session_id", "selected_id"]], on="session_id", how="left")
    raw["y"] = (raw["DiscomfortLevel"] >= POSITIVE_FROM_GRADE).astype(int)
    raw["grade"] = raw["DiscomfortLevel"].astype(int)

    sessions = session_table(raw)
    eval_mask = raw["selected_id"].notna()
    dev_mask = raw["wave"].str.startswith("2019-10")

    groups_eval = linkage_groups(sessions[sessions.session_id.isin(raw.loc[eval_mask, "session_id"])])
    groups_dev = linkage_groups(sessions[sessions.session_id.isin(raw.loc[dev_mask, "session_id"])])
    groups_dev["group_id"] = "D" + groups_dev["group_id"].str[1:]

    ev = raw[eval_mask].merge(groups_eval[["session_id", "group_id"]], on="session_id")
    dv = raw[dev_mask].merge(groups_dev[["session_id", "group_id"]], on="session_id")
    ev = ev.sort_values(["selected_id", "step"]).reset_index(drop=True)
    dv = dv.sort_values(["start_time", "step"]).reset_index(drop=True)

    sessions = sessions.merge(pd.concat([groups_eval, groups_dev])[["session_id", "group_id"]],
                              on="session_id", how="left")
    sessions["role"] = np.select(
        [sessions.session_id.isin(ev.session_id), sessions.session_id.isin(dv.session_id)],
        ["evaluation (curated release)", "development only"], default="unused (not in curated release)")
    sessions = sessions.merge(link[["session_id", "selected_id"]], on="session_id", how="left")
    sessions.attrs["link_log_eval"] = groups_eval.attrs["link_log"]
    sessions.attrs["link_log_dev"] = groups_dev.attrs["link_log"]
    return ev, dv, sessions


def _strat_label(long: pd.DataFrame) -> np.ndarray:
    return (2 * (long["game"] == "Flight").astype(int) + long["y"]).to_numpy()


def grouped_folds(long: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """Assign every linkage group to exactly one fold (stratified on game x label)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = {}
    for k, (_, te) in enumerate(sgkf.split(long, _strat_label(long), long["group_id"])):
        for gid in long.iloc[te]["group_id"].unique():
            fold_of[gid] = k
    out = (long.groupby("group_id")
           .agg(sessions=("session_id", lambda s: ";".join(sorted(s.unique()))),
                n_sessions=("session_id", "nunique"), n_rows=("y", "size"), n_pos=("y", "sum"),
                games=("game", lambda s: "/".join(sorted(s.unique()))))
           .reset_index())
    out["fold"] = out["group_id"].map(fold_of)
    return out.sort_values(["fold", "group_id"]).reset_index(drop=True)


# ------------------------------------------------------------- splitting
def split_tables(long: pd.DataFrame, protocol: dict) -> dict:
    outer = grouped_folds(long, protocol["k_outer"], protocol["split_seed"])
    tables = {"outer": outer, "inner": {}}
    for k in range(protocol["k_outer"]):
        dev = long[long.group_id.isin(outer.loc[outer.fold != k, "group_id"])]
        tables["inner"][k] = grouped_folds(dev, protocol["k_inner"], protocol["split_seed"] + 1 + k)
    return tables


def fold_frames(long, tables, outer_k, inner_j=None):
    outer = tables["outer"]
    test = long[long.group_id.isin(outer.loc[outer.fold == outer_k, "group_id"])]
    dev = long[~long.group_id.isin(test.group_id)]
    if inner_j is None:
        return dev, test
    inner = tables["inner"][outer_k]
    val = dev[dev.group_id.isin(inner.loc[inner.fold == inner_j, "group_id"])]
    tr = dev[~dev.group_id.isin(val.group_id)]
    return tr, val, test


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


@dataclass
class FoldData:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_ev: dict            # name -> matrix (validation / test), transformed with frozen params
    y_ev: dict
    meta_tr: pd.DataFrame  # session_id, step, group_id, y, grade (row order of X_tr)
    meta_ev: dict
    feature_names: list
    params: dict          # learned preprocessing parameters (train-only)


META_COLS = ["session_id", "group_id", "game", "step", "TimeStamp", "y", "grade"]


def prepare_fold(train_long: pd.DataFrame, eval_longs: dict, with_profile: bool = False) -> FoldData:
    """Fit featurizer + scaler on training rows only, apply frozen to the rest."""
    overlap = {k: set(train_long.session_id) & set(v.session_id) for k, v in eval_longs.items()}
    goverlap = {k: set(train_long.group_id) & set(v.group_id) for k, v in eval_longs.items()}
    if any(overlap.values()) or any(goverlap.values()):
        raise RuntimeError(f"session/group overlap between partitions: {overlap} {goverlap}")
    fz = CausalFeaturizer(with_profile=with_profile).fit(train_long)
    F_tr = fz.transform(train_long)
    scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(F_tr.to_numpy(float))
    X_tr = scaler.transform(F_tr.to_numpy(float)).astype(np.float32)
    X_ev, y_ev, meta_ev = {}, {}, {}
    for name, lg in eval_longs.items():
        F = fz.transform(lg)
        X_ev[name] = scaler.transform(F.to_numpy(float)).astype(np.float32)
        y_ev[name] = lg["y"].to_numpy()
        meta_ev[name] = lg[META_COLS].reset_index(drop=True)
    params = fz.learned_params()
    params["scaler_center"] = dict(zip(F_tr.columns, scaler.center_.round(6).tolist()))
    params["scaler_scale"] = dict(zip(F_tr.columns, scaler.scale_.round(6).tolist()))
    return FoldData(X_tr, train_long["y"].to_numpy(), X_ev, y_ev,
                    train_long[META_COLS].reset_index(drop=True), meta_ev, list(F_tr.columns), params)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def processed_path(name: str):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    return PROCESSED / name
