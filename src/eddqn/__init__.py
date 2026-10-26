"""Leakage-free rebuild of the E-DDQN cybersickness-detection study.

Every module in this package reads the original, unmodified UFFCSData XML
files from ``UFFCSData-master_raw_source`` and never touches the legacy
processed CSVs used by the original manuscript (which are not in the
repository and whose preprocessing history cannot be verified).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_ROOT = REPO_ROOT
RAW_ROOT = REPO_ROOT / "UFFCSData-master_raw_source"
PROCESSED = REVISION_ROOT / "data" / "processed"
RESULTS = REVISION_ROOT / "results"
CONFIGS = REVISION_ROOT / "configs"
