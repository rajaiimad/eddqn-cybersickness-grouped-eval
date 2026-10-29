#!/usr/bin/env bash
# Regenerate every experimental result from the raw UFFCSData XML files.
# Usage: bash run_all.sh [--workers N] [--fresh]
#   --fresh  delete every generated result (results/ and the split tables) first.
# Wall time ~2.5-3.5 h on 4 cores (dominated by the RL hyper-parameter searches and ablation).
set -euo pipefail
cd "$(dirname "$0")"
WORKERS=4; FRESH=0
while [[ $# -gt 0 ]]; do case $1 in --workers) WORKERS=$2; shift 2;; --fresh) FRESH=1; shift;; *) echo "unknown $1"; exit 1;; esac; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
if [[ $FRESH == 1 ]]; then   # remove every generated result
  rm -rf results/* data/processed/*
  mkdir -p results data/processed
fi
python3 scripts/00_sanity_tests.py                      # unit checks of the corrected agent
python3 scripts/01_audit_original.py                    # provenance, leakage reproduction
bash    scripts/01b_run_original_proxy.sh               # unmodified earlier script on proxy splits
python3 scripts/02_preprocessing_audit.py               # splits + leakage assertions (writes data/processed)
python3 scripts/dev_calibration.py                      # development-wave calibration (Oct-2019 only)
FAM="E-DDQN E-CB DQN CS-MLP LogReg HGB RF"
python3 scripts/03_run_protocol.py hpo    $FAM --workers "$WORKERS"
python3 scripts/03_run_protocol.py select $FAM
python3 scripts/03_run_protocol.py final  $FAM --workers "$WORKERS"
python3 scripts/03_run_protocol.py ablation --workers "$WORKERS"
python3 scripts/03_run_protocol.py ratio    --workers "$WORKERS"
python3 scripts/04_aggregate.py                          # metrics, CIs, paired tests, figures
python3 scripts/07_shap.py                               # SHAP attributions
python3 scripts/05_make_tables.py                        # result tables and macros (results/tables/latex*)
python3 scripts/11_posthoc.py run --workers "$WORKERS"  # post-hoc baselines (configs/posthoc.json)
python3 scripts/11_posthoc.py analyze                    # post-hoc statistics, tables, figures
python3 scripts/13_search_draw_sensitivity.py run --workers "$WORKERS"  # search-draw sensitivity (configs/posthoc.json)
python3 scripts/13_search_draw_sensitivity.py analyze    # its statistics and decision rules
python3 scripts/14_score_scale.py                        # score scale of rule D0 (descriptive, no training)
python3 scripts/15_pipeline_verification.py              # split, preprocessing, selection and replay checks on the real folds
python3 scripts/16_threshold_scale.py                    # threshold-scale sensitivity of rule D1 (post hoc, no training)
python3 scripts/19_all_family_draws.py run --workers "$WORKERS"  # draws 2-3 of the remaining families (configs/posthoc.json)
python3 scripts/19_all_family_draws.py analyze            # draw-matched and draw-averaged comparisons, MDE, seed-mean tests
python3 scripts/20_inputs_checkpoint_events.py run --workers "$WORKERS"  # exposure-only for every family, last-epoch retraining
python3 scripts/20_inputs_checkpoint_events.py analyze              # inputs-fixed comparisons, checkpoint check, event-level metrics
