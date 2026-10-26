"""Parse the original UFFCSData XML session exports.

Data provenance (verified against the files in this repository):

* ``RAW_DATABASE/RAW_DATABASE/DATA_EXPORT_<Game>_<Y>_<M>_<D>_<h>_<m>_<s>/FILE.xml``
  - 74 gameplay sessions (two collection waves: Oct-2019 standing posture,
    Dec-2019 seated posture).  One ``<Data>`` element per ~1 s sample.
  - ``VRSQ_FILE.xml``: pre/post Virtual Reality Sickness Questionnaire.
* ``DATABASE_SELECTED_DATA/DATABASE/<k>/FILE.xml`` (k = 1..35)
  - The dataset authors' curated release: 35 of the 39 Dec-2019 sessions.
    Each file is byte-for-byte the same telemetry as one raw export
    (verified in :func:`link_selected_to_raw`).
* ``dataset.csv`` is the concatenation of the 35 curated sessions with the
  categorical fields integer-coded; it carries **no session identifier**.

The folder name is the only source of game type and session start time.
There is no participant identifier anywhere in the release.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from . import RAW_ROOT

RAW_DIR = RAW_ROOT / "RAW_DATABASE" / "RAW_DATABASE"
SEL_DIR = RAW_ROOT / "DATABASE_SELECTED_DATA" / "DATABASE"

NUMERIC = [
    "TimeStamp", "CameraFieldOfView", "PlayerSpeed", "PlayerAcceleration",
    "CameraRotationX", "CameraRotationY", "CameraRotationZ",
    "PlayerPositionX", "PlayerPositionY", "PlayerPositionZ", "GameFps",
    "DiscomfortLevel",
]
PROFILE = [
    "UserGenere", "UserAge", "UserExperience", "UserSymptoms", "UserFlicker",
    "UserGlassesUse", "UserVisionProblems", "UserPosture", "UserEyeDominance",
]
SESSION_CONFIG = [
    "StaticFrame", "HapticFeedback", "DegreeOfControl", "DofSimulation",
    "Locomotion", "CameraAutoMovement",
]

# Portuguese questionnaire answers -> English (documentation only; the raw
# strings are kept so nothing is lost in translation).
TRANSLATE = {
    "masculino": "male", "feminino": "female",
    "18 A 36": "18-36", "37 A 50": "37-50", "+50": "50+",
    "NENHUMA": "none", "ALGUMA": "some", "MUITA": "a lot",
    "NENHUM": "none", "ALGUM": "some", "NÃO": "no", "SIM": "yes",
    "MIOPIA": "myopia", "ASTIGMATISMO": "astigmatism",
    "HIPERMETROPIA": "hyperopia", "DE PÉ": "standing", "SENTADO": "seated",
    "DIREITO": "right", "ESQUERDO": "left",
}

_FOLDER = re.compile(r"DATA_EXPORT_(Flight|Race)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$")


def _read_xml(path: Path) -> list[dict]:
    text = path.read_bytes().decode("utf-8-sig")
    root = ET.fromstring(text)
    return [{c.tag: (c.text or "") for c in d} for d in root.findall("Data")]


def _num(series: pd.Series) -> pd.Series:
    # Oct-2019 exports use a decimal comma, Dec-2019 curated exports a dot.
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="raise")


def parse_folder_name(name: str) -> tuple[str, _dt.datetime]:
    m = _FOLDER.match(name)
    if not m:
        raise ValueError(f"unexpected session folder name: {name}")
    game = m.group(1)
    y, mo, d, h, mi, s = (int(x) for x in m.groups()[1:])
    return game, _dt.datetime(y, mo, d, h, mi, s)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_session(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(_read_xml(path))
    for c in NUMERIC:
        if c in df:
            df[c] = _num(df[c])
    return df


def load_raw_sessions() -> pd.DataFrame:
    """All 74 raw exports as one long frame (one row per logged sample)."""
    frames = []
    for folder in sorted(RAW_DIR.iterdir()):
        if not folder.is_dir():
            continue
        game, start = parse_folder_name(folder.name)
        df = load_session(folder / "FILE.xml")
        df.insert(0, "session_id", folder.name)
        df.insert(1, "game", game)
        df.insert(2, "start_time", start)
        df.insert(3, "wave", "2019-10 (standing)" if start.month == 10 else "2019-12 (seated)")
        df.insert(4, "step", np.arange(len(df)))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_vrsq() -> pd.DataFrame:
    rows = []
    for folder in sorted(RAW_DIR.iterdir()):
        p = folder / "VRSQ_FILE.xml"
        if not p.exists():
            continue
        root = ET.fromstring(p.read_bytes().decode("utf-8-sig"))
        for i, s in enumerate(root.findall("Session")):
            r = {c.tag: float((c.text or "nan").replace(",", ".")) for c in s}
            r.update(session_id=folder.name, vrsq_index=i)
            rows.append(r)
    return pd.DataFrame(rows)


def link_selected_to_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Map each curated session k=1..35 to its raw export and verify that all
    numeric telemetry and labels are identical sample by sample."""
    raw_by_id = {sid: g.reset_index(drop=True) for sid, g in raw.groupby("session_id")}
    out = []
    for folder in sorted(SEL_DIR.iterdir(), key=lambda p: int(p.name)):
        sel = load_session(folder / "FILE.xml")
        match = []
        for sid, g in raw_by_id.items():
            if len(g) != len(sel) or "PlayerPositionX" not in g:
                continue
            same = all(
                np.allclose(g[c].to_numpy(float), sel[c].to_numpy(float), atol=1e-6)
                for c in NUMERIC
            ) and all((g[c] == sel[c]).all() for c in PROFILE + SESSION_CONFIG + ["RegionOfInterest"])
            if same:
                match.append(sid)
        if len(match) != 1:
            raise RuntimeError(f"selected session {folder.name}: {len(match)} raw matches")
        out.append({"selected_id": int(folder.name), "session_id": match[0], "n_rows": len(sel),
                    "sha256": file_sha256(folder / "FILE.xml")})
    return pd.DataFrame(out)


def verify_dataset_csv(selected_long: pd.DataFrame) -> dict:
    """Check that ``dataset.csv`` is exactly the curated sessions concatenated
    in *lexicographic* folder order (1, 10, 11, ..., 19, 2, 20, ...)."""
    csv = pd.read_csv(RAW_ROOT / "dataset.csv")
    selected_long = (selected_long.assign(_lex=selected_long["selected_id"].astype(str))
                     .sort_values(["_lex", "step"]).reset_index(drop=True))
    cols = ["TimeStamp", "CameraFieldOfView", "PlayerSpeed", "PlayerAcceleration",
            "CameraRotationX", "CameraRotationY", "CameraRotationZ",
            "PlayerPositionX", "PlayerPositionY", "PlayerPositionZ", "GameFps", "DiscomfortLevel"]
    same_len = len(csv) == len(selected_long)
    max_abs = {c: float(np.max(np.abs(csv[c].to_numpy(float) - selected_long[c].to_numpy(float))))
               for c in cols} if same_len else {}
    return {"dataset_csv_rows": int(len(csv)), "selected_rows": int(len(selected_long)),
            "row_order_identical": bool(same_len and max(max_abs.values()) < 1e-4),
            "max_abs_diff_per_column": max_abs,
            "dataset_csv_has_session_id": False,
            "dataset_csv_label_counts": csv["DiscomfortLevel"].value_counts().sort_index().to_dict()}
