"""Causal (prospective) feature engineering.

Every feature of sample t in session s is a function of samples 0..t of the
same session only (trailing windows, cumulative sums).  Data-driven constants
(the acceleration-spike threshold and the imputation value) are learned in
:meth:`CausalFeaturizer.fit` from TRAINING rows only and frozen afterwards.

Removed relative to the raw export (see results/preprocessing/data_quality.json):

* ``DiscomfortLevel`` and anything derived from it (label).
* ``CameraFieldOfView``: constant within each game in the curated release
  (Race 96.016, Flight 111.755) -> a duplicate of the game indicator.
* ``StaticFrame``, ``CameraAutoMovement``: constant within each game
  (Race: True/False, Flight: False/True) -> duplicates of the game indicator.
* ``HapticFeedback``, ``DegreeOfControl``, ``DofSimulation``, ``Locomotion``,
  ``UserPosture``: constant over the whole curated release.
* ``PlayerPositionX/Y/Z``: absolute course coordinates; Z grows almost
  linearly with time in the Flight game, the columns are absent from the
  Oct-2019 wave, and they would let a model memorise course locations.
Participant questionnaire fields (gender, age band, VR experience, glasses,
vision problems, eye dominance, flicker) are filled in BEFORE the session
(Cybersickness Profile Questionnaire), so they are prospective.  They are
constant per participant and could act as quasi-identifiers under a
row-level split; under the linkage-group split used here a test group's
exact profile never occurs in its training data (groups are built from
identical profiles), so they can only contribute through shared attribute
values (e.g. "no prior VR experience").  Whether to use them is a tuned
feature-set choice made on inner validation folds only; ``UserSymptoms`` and
``UserPosture`` are constant in the curated release and are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (5, 15)          # seconds (samples are logged at ~1 Hz)
FPS_DROP_BELOW = 80.0      # fixed physical threshold: >11 % below the 90-Hz HMD target
ACC_SPIKE_QUANTILE = 0.90  # learned on training rows only
ROI_LEVELS = ("background", "middleground", "foreground")

PROFILE_ONEHOT = {
    "UserGenere": ["masculino", "feminino"],
    "UserAge": ["18 A 36", "37 A 50", "+50"],
    "UserExperience": ["NENHUMA", "ALGUMA", "MUITA"],
    "UserGlassesUse": ["NÃO", "SIM"],
    "UserEyeDominance": ["DIREITO", "ESQUERDO"],
    "UserFlicker": ["NENHUM", "ALGUM"],
}

# Feature sets offered to EVERY model family as a tuned hyper-parameter
# (chosen by inner-CV validation only).  "exposure" is a small, a-priori set of
# cumulative/trailing exposure descriptors motivated by the cybersickness
# literature (exposure duration, accumulated rotation/acceleration).
EXPOSURE = ["is_flight", "elapsed_s", "cum_rot", "cum_acc_spikes", "speed_mean_sofar",
            "speed_mean_15", "rotrate_mean_15", "absacc_mean_15"]


def feature_columns(all_names: list, feature_set: str) -> list:
    prof = [c for c in all_names if c.startswith("prof_")]
    tele = [c for c in all_names if not c.startswith("prof_")]
    sets = {"telemetry": tele, "telemetry+profile": tele + prof,
            "exposure": EXPOSURE, "exposure+profile": EXPOSURE + prof,
            # post-hoc exposure-time baselines (configs/posthoc.json); never
            # offered to the frozen families
            "time": ["elapsed_s"], "time+game": ["elapsed_s", "is_flight"]}
    return sets[feature_set]


# Raw channels an adversary could plausibly influence through the rendering /
# locomotion pipeline (used by attacks.py).  Elapsed time, game and region of
# interest are not treated as attackable.
MUTABLE_RAW = ("PlayerSpeed", "PlayerAcceleration", "CameraRotationX",
               "CameraRotationY", "CameraRotationZ", "GameFps")


class CausalFeaturizer:
    def __init__(self, with_profile: bool = False):
        self.with_profile = with_profile
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, train_long: pd.DataFrame) -> "CausalFeaturizer":
        fps = self._clean_fps(train_long)
        self.fps_impute_ = float(np.nanmedian(fps))
        self.acc_spike_thr_ = float(np.quantile(np.abs(train_long["PlayerAcceleration"]), ACC_SPIKE_QUANTILE))
        self.train_sessions_ = sorted(train_long["session_id"].unique())
        self.fitted_ = True
        return self

    def learned_params(self) -> dict:
        return {"fps_impute_median": self.fps_impute_,
                "acc_spike_threshold_abs": self.acc_spike_thr_,
                "fps_drop_threshold_fixed": FPS_DROP_BELOW,
                "fitted_on_n_sessions": len(self.train_sessions_)}

    @staticmethod
    def _clean_fps(long: pd.DataFrame) -> pd.Series:
        # The Unity logger writes GameFps = 0 on the very first sample of some
        # sessions (TimeStamp < 0.02 s): a start-up artefact, not a frame drop.
        fps = long["GameFps"].astype(float).copy()
        fps[(fps <= 0) & (long["TimeStamp"] < 1.0)] = np.nan
        return fps

    # ------------------------------------------------------------ transform
    def transform(self, long: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("fit on training rows first")
        parts = [self._session_features(g) for _, g in long.groupby("session_id", sort=False)]
        feats = pd.concat(parts)
        return feats.loc[long.index]

    def _session_features(self, g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("TimeStamp")
        f = pd.DataFrame(index=g.index)
        f["is_flight"] = (g["game"] == "Flight").astype(float)
        f["elapsed_s"] = g["TimeStamp"].astype(float)
        f["speed"] = g["PlayerSpeed"].astype(float)
        f["accel"] = g["PlayerAcceleration"].astype(float)
        absacc = f["accel"].abs()
        f["abs_accel"] = absacc
        q = g[["CameraRotationX", "CameraRotationY", "CameraRotationZ"]].astype(float)
        f["rot_x"], f["rot_y"], f["rot_z"] = q.iloc[:, 0], q.iloc[:, 1], q.iloc[:, 2]
        rot_rate = np.sqrt((q.diff() ** 2).sum(axis=1))
        rot_rate.iloc[0] = 0.0
        f["rot_rate"] = rot_rate
        fps = self._clean_fps(g).fillna(self.fps_impute_)
        f["fps"] = fps
        for lvl in ROI_LEVELS:
            f[f"roi_{lvl}"] = (g["RegionOfInterest"] == lvl).astype(float)

        spikes = (absacc > self.acc_spike_thr_).astype(float)
        drops = (fps < FPS_DROP_BELOW).astype(float)
        for w in WINDOWS:
            r = lambda s: s.rolling(w, min_periods=1)  # trailing window incl. t
            f[f"speed_mean_{w}"] = r(f["speed"]).mean()
            f[f"speed_std_{w}"] = r(f["speed"]).std().fillna(0.0)
            f[f"speed_delta_{w}"] = f["speed"] - f["speed"].shift(w - 1).fillna(f["speed"].iloc[0])
            f[f"absacc_mean_{w}"] = r(absacc).mean()
            f[f"absacc_max_{w}"] = r(absacc).max()
            f[f"acc_spike_rate_{w}"] = r(spikes).mean()
            f[f"rotrate_mean_{w}"] = r(rot_rate).mean()
            f[f"rotrate_std_{w}"] = r(rot_rate).std().fillna(0.0)
            f[f"fps_mean_{w}"] = r(fps).mean()
            f[f"fps_min_{w}"] = r(fps).min()
            f[f"fps_drop_frac_{w}"] = r(drops).mean()
        n = np.arange(1, len(g) + 1)
        f["cum_rot"] = rot_rate.cumsum()
        f["cum_acc_spikes"] = spikes.cumsum()
        f["cum_fps_drops"] = drops.cumsum()
        f["speed_mean_sofar"] = f["speed"].cumsum() / n

        if self.with_profile:
            first = g.iloc[0]
            for col, levels in PROFILE_ONEHOT.items():
                for lv in levels[1:]:
                    f[f"prof_{col}_{lv}"] = float(first[col] == lv)
            vp = str(first["UserVisionProblems"])
            for cond in ("MIOPIA", "ASTIGMATISMO", "HIPERMETROPIA"):
                f[f"prof_vision_{cond}"] = float(cond in vp)
        return f



_PROFILE_NAMES = {
    "UserGenere_feminino": "gender: female", "UserAge_37 A 50": "age 37-50", "UserAge_+50": "age 50+",
    "UserExperience_ALGUMA": "VR experience: some", "UserExperience_MUITA": "VR experience: a lot",
    "UserGlassesUse_SIM": "wears glasses", "UserEyeDominance_ESQUERDO": "left-eye dominant",
    "UserFlicker_ALGUM": "flicker sensitivity: some", "vision_MIOPIA": "vision: myopia",
    "vision_ASTIGMATISMO": "vision: astigmatism", "vision_HIPERMETROPIA": "vision: hyperopia",
}


def display_name(feature: str) -> str:
    """Human-readable English label (questionnaire answers are stored in Portuguese)."""
    if feature.startswith("prof_"):
        return "CSPQ " + _PROFILE_NAMES.get(feature[5:], feature[5:])
    return feature
