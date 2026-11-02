#!/usr/bin/env bash
# Run the UNMODIFIED submitted script (DRL_cybersickness_FAST_v2.py) on the two
# proxy splits written by 01_audit_original.py:
#   leaky_oversampled : random oversampling to 5,243/grade, then a 70/15/15
#                       row split (reproduces 14,680 / 3,146 / 3,146 exactly)
#   grouped           : the same columns, linkage-group split (no oversampling)
# The exact 41-feature CSVs of the manuscript are not in the repository, so
# this is a proxy reproduction, not an exact one.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROOT="$HERE/code/original"
for v in leaky_oversampled grouped; do
  RUN="$HERE/results/original_repro/run/$v"
  mkdir -p "$RUN"
  cp "$ROOT/DRL_cybersickness_FAST_v2.py" "$RUN/"
  cp "$HERE/results/audit/proxy_original_splits/$v/"*.csv "$RUN/"
  ( cd "$RUN" && OMP_NUM_THREADS=1 MPLBACKEND=Agg python3 DRL_cybersickness_FAST_v2.py > run.log 2>&1 || echo "exit $?" >> run.log )
  mkdir -p "$HERE/results/original_repro/$v"
  cp "$RUN/run.log" "$HERE/results/original_repro/$v/run.log"
  cp "$RUN"/Results_FastDQL/*.xlsx "$RUN"/Results_FastDQL/*.csv "$HERE/results/original_repro/$v/" 2>/dev/null || true
  python3 -c "import pandas as pd,sys; pd.read_excel(sys.argv[1]).to_csv(sys.argv[2], index=False)" \
    "$RUN/Results_FastDQL/Split_Comparison_FastDQL.xlsx" "$HERE/results/original_repro/$v/split_comparison.csv" || true
done
