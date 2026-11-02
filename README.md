# E-DDQN cybersickness detection: session-grouped evaluation code and data

Research code and data for the study *"Does an Asymmetric Reward Improve a Double DQN for Cybersickness Detection? A Session-Grouped Re-evaluation on the UFFCSData Gameplay Dataset"* (R. Darabee, M. Muniandy, O. Shindi, C. W. Khuen, E. H. Sumiea).

The code regenerates every experimental result from the raw UFFCSData session files. The saved per-sample predictions and result tables let you check the reported numbers without retraining.

## Structure

```
code/
  src/eddqn/        raw-data parser, session-linkage grouping, causal features, E-DDQN and
                    baseline models (DQN, E-CB, CS-MLP, LogReg, HGB, RF), metrics, statistics,
                    telemetry perturbations
  scripts/          pipeline steps 00-22, run in order by run_all.sh
  configs/          protocol.json (fixed before evaluation), posthoc.json (post-hoc analyses and
                    whether each was specified before it was run), attacks.json (perturbations),
                    FROZEN_PROTOCOL_COMMIT.txt
  original/         DRL_cybersickness_FAST_v2.py: the earlier implementation, run unmodified
                    only to reconstruct the earlier (leaky) protocol
  run_all.sh        regenerates everything
  requirements.txt  pinned package versions
data/
  raw/UFFCSData-master_raw_source/   UFFCSData files used by the pipeline (see Data)
  processed/        outer and inner split tables (session-linkage groups), perturbation masks
results/            outputs of the pipeline (CSV / JSON)
  predictions/      per-sample predictions of every model, seed and outer fold (*.csv.gz)
  tables/           metrics, confidence intervals, paired tests, ablation, reward ratio,
                    perturbations, post-hoc, search-draw and reporting summaries
  preprocessing/, audit/, qc/, dev_calibration/, original_repro/
                    data audit, split and leakage checks, reproduction of the earlier protocol,
                    pipeline verification, development-wave calibration
  hpo_results*.csv, selected_configs*.json   hyper-parameter searches and selected configurations
```

## Reproduce

Requirements: Python 3.13 and the packages in `code/requirements.txt`. A CPU is sufficient. Run from the repository root:

```bash
pip install -r code/requirements.txt
bash code/run_all.sh --fresh --workers 4   # about 2.5-3.5 h on 4 CPU cores
```

`--fresh` deletes every generated result and rebuilds everything from the raw XML files:

1. Sanity tests of the agents (`00`).
2. Provenance audit and reconstruction of the earlier, leaky protocol (`01`, `01b`).
3. Preprocessing, splits and leakage assertions (`02`), development-wave calibration (`dev_calibration.py`).
4. Hyper-parameter search, selection, final evaluation, ablation and reward-ratio runs (`03`).
5. Aggregation, statistics and SHAP (`04`, `07`), result tables (`05`).
6. Post-hoc baselines (`11`), search-draw sensitivity (`13`), three search draws for every model family (`19`), exposure-only searches, last-epoch retraining and event-level metrics (`20`).
7. Score-scale, pipeline-verification, threshold-scale and reporting checks (`14`, `15`, `16`, `21`, `22`).

Quick checks that need no training (minutes):

```bash
python3 code/scripts/00_sanity_tests.py          # agent unit checks
python3 code/scripts/02_preprocessing_audit.py   # rebuilds data/processed and the leakage checks
python3 code/scripts/22_reporting_summaries.py   # summaries from the saved result tables
```

The trained-model files (`results/runs/`, about 300 MB) are not included; `run_all.sh` regenerates them. Steps `04`, `14`, `15` and `16` read them, so they need a full run first. The scripts also write figures, LaTeX tables and logs; these are not part of the repository (see `.gitignore`).

Seeds: data splitting 20261012 (outer) and 20261013-20261017 (inner); hyper-parameter search 777 (801-802 for the post-hoc baselines; 21 configuration seeds between 9120 and 9208 for the further search draws, set in `code/configs/posthoc.json`); final training 1000-1004; perturbations 4242. Every process runs on CPU with one thread, so results are deterministic on a given platform; minor floating-point differences across hardware are possible.

## Design

* **Data:** the 35-session curated UFFCSData release is used for evaluation; the 37 October-2019 sessions are used only for development.
* **Unit of independence:** session-linkage groups reconstructed from questionnaire profiles and session timing, because the release has no participant identifiers. The checks establish session- and group-disjoint evaluation, not participant independence.
* **Splits:** nested grouped cross-validation (5 outer x 3 inner folds).
* **Features:** causal features only; all learned preprocessing is fitted on training groups.
* **Comparison:** every model gets the same search procedure and budget, checkpoint rule, cross-fitting recipe, operating rules and paired statistics.
* **Primary decision rule:** the threshold minimising the 5:1 cost on validation predictions.
* **Inference:** group bootstrap (primary) and exact group sign-flip test (sensitivity), with Holm adjustment.

The protocol and the post-hoc specifications were committed, with timestamps, in the authors' development repository before the corresponding runs; this repository is a curated copy.

## Data

UFFCSData was collected and released by Porcino et al. (<https://github.com/tmp1986/UFFCSData>) under the GNU AGPL-3.0; its licence is in `data/raw/UFFCSData-master_raw_source/LICENSE`. Included here:

* the curated release (`DATABASE_SELECTED_DATA`);
* the raw exports (`RAW_DATABASE`);
* `dataset.csv`, used only for the reconstruction of the earlier protocol.

The game build and the Weka files of the original release are not included; they are available from the original repository.

## Licence

Code: to be chosen by the authors. The included UFFCSData files remain under AGPL-3.0.
