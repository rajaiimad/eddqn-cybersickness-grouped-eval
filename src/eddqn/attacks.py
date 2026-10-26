"""Bounded telemetry-channel perturbations (synthetic feature-space stress tests).

Perturbations are applied to the RAW per-second channels of a session, and the
causal features are then recomputed with the frozen, training-fitted
featurizer and scaler of the fold - exactly as a deployed monitor would see
manipulated telemetry.  The cybersickness label y is never modified; the
attack indicator z marks perturbed samples.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from . import CONFIGS

SPEED_RANGE = {"Flight": (10.0, 130.0), "Race": (0.0, 146.0)}
ROT = ["CameraRotationX", "CameraRotationY", "CameraRotationZ"]


def load_attack_config() -> dict:
    return json.loads((CONFIGS / "attacks.json").read_text())


def _recompute_accel(g: pd.DataFrame) -> pd.Series:
    acc = g["PlayerSpeed"].diff()
    acc.iloc[0] = g["PlayerAcceleration"].iloc[0]
    return acc


def _window_mask(n: int, window: str, rng) -> np.ndarray:
    m = np.zeros(n, bool)
    if window == "whole":
        m[:] = True
    else:
        w = 60
        start = int(rng.integers(0, max(n - w, 0) + 1))
        m[start:start + w] = True
    return m


def perturb_session(g: pd.DataFrame, kind: str, value: float, window: str, rng) -> tuple[pd.DataFrame, np.ndarray]:
    g = g.sort_values("TimeStamp").copy()
    n = len(g)
    m = _window_mask(n, window, rng)
    lo, hi = SPEED_RANGE[g["game"].iloc[0]]
    if kind == "fps_cap":
        fps = g["GameFps"].to_numpy(float)
        start_art = (fps <= 0) & (g["TimeStamp"].to_numpy() < 1.0)   # keep the logger artefact
        fps = np.where(m & ~start_art, np.minimum(fps, value), fps)
        g["GameFps"] = fps
    elif kind == "fps_restore":
        fps = g["GameFps"].to_numpy(float)
        start_art = (fps <= 0) & (g["TimeStamp"].to_numpy() < 1.0)
        g["GameFps"] = np.where(m & ~start_art, value, fps)
    elif kind == "rot_jitter":
        for c in ROT:
            noise = rng.uniform(-value, value, n)
            g[c] = np.clip(g[c].to_numpy(float) + np.where(m, noise, 0.0), -1, 1)
    elif kind == "rot_smooth":
        q = g[ROT].to_numpy(float)
        out = q.copy()
        for t in range(1, n):
            out[t] = out[t - 1] + value * (q[t] - q[t - 1]) if m[t] else q[t]
        g[ROT] = np.clip(out, -1, 1)
    elif kind in ("speed_scale_up", "speed_scale_down"):
        sp = g["PlayerSpeed"].to_numpy(float)
        g["PlayerSpeed"] = np.where(m, np.clip(sp * (1 + value), lo, hi), sp)
        g["PlayerAcceleration"] = _recompute_accel(g)
    elif kind == "accel_spikes":
        sp = g["PlayerSpeed"].to_numpy(float)
        bump = np.zeros(n)
        idx = np.flatnonzero(m)
        bump[idx[::20]] = value
        g["PlayerSpeed"] = np.clip(sp + bump, lo, hi)
        g["PlayerAcceleration"] = _recompute_accel(g)
    else:
        raise ValueError(kind)
    return g, m


def variants(cfg: dict) -> list[dict]:
    """All perturbation variants (frozen list, deterministic order)."""
    out = []
    for kind, spec in cfg["stress_scenarios"].items():
        for v in spec["values"]:
            for w in cfg["windows"]:
                out.append({"name": f"{kind}={v}@{w}", "kind": kind, "value": float(v), "window": w,
                            "role": "stress"})
    for kind, spec in cfg["evasion_candidates"].items():
        if kind.startswith("_"):
            continue
        for v in spec["values"]:
            out.append({"name": f"{kind}={v}@whole", "kind": kind, "value": float(v), "window": "whole",
                        "role": "evasion_only"})
    return out


def perturb_long(long: pd.DataFrame, var: dict, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Perturb every session of ``long``; returns frame in the ORIGINAL row
    order and the attack indicator z."""
    parts, masks = [], []
    for k, (sid, g) in enumerate(long.groupby("session_id", sort=False)):
        rng = np.random.default_rng([seed, k, zlib.crc32(var["name"].encode())])
        pg, m = perturb_session(g, var["kind"], var["value"], var["window"], rng)
        parts.append(pg)
        masks.append(pd.Series(m, index=pg.index))
    out = pd.concat(parts).loc[long.index]
    z = pd.concat(masks).loc[long.index].to_numpy()
    return out, z
