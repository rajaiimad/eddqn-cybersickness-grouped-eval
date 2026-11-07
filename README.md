# E-DDQN Cybersickness Detection: Session-Grouped Evaluation Code and Data

This repository contains the code and data for the following manuscript:

| | |
|---|---|
| **Title** | Does an asymmetric reward improve a Double DQN for cybersickness detection? A session-grouped re-evaluation on the UFFCSData gameplay dataset |
| **Authors** | Rajai Darabee, Manoranjitham Muniandy, Wai Khuen Cheng, Ebrahim Hamid Sumiea, Omar Shindi |
| **Journal** | *PeerJ Computer Science* |
| **Status** | Submitted, under review |

The study tests whether E-DDQN, a dueling Double Deep Q-Network whose reward penalizes a missed cybersickness event five times more than a false alarm, detects cybersickness better than a standard DQN and cost-weighted supervised models once data leakage is controlled. All models are evaluated under the same nested, session-grouped protocol on the public UFFCSData gameplay dataset.

The repository allows reviewers and readers to:

- **check every reported number** from the saved per-sample predictions and result tables, without retraining; and
- **regenerate every result** from the raw UFFCSData session files with a single script.

## Repository structure

```
code/
  src/eddqn/         Raw-data parser, session-linkage grouping, causal features, E-DDQN and the
                     baseline models (DQN, E-CB, CS-MLP, LogReg, HGB, RF), metrics, statistics
                     and telemetry perturbations
  scripts/           Pipeline steps 00-22, run in order by run_all.sh
  configs/           protocol.json (evaluation protocol, fixed before evaluation),
                     posthoc.json (post-hoc analyses, stating whether each was specified
                     before it was run), attacks.json (perturbation scenarios),
                     FROZEN_PROTOCOL_COMMIT.txt
  original/          DRL_cybersickness_FAST_v2.py: the earlier implementation, run unmodified
                     only to reconstruct the earlier (leaky) evaluation protocol
  run_all.sh         Regenerates all results
  requirements.txt   Pinned package versions
data/
  raw/UFFCSData-master_raw_source/   UFFCSData files used by the pipeline (see Data)
  processed/         Outer and inner split tables (session-linkage groups), perturbation masks
results/             Pipeline outputs (CSV / JSON)
  predictions/       Per-sample predictions of every model, seed and outer fold (*.csv.gz)
  tables/            Metrics, confidence intervals, paired tests, ablation, reward ratio,
                     perturbations, post-hoc, search-draw and reporting summaries
  preprocessing/, audit/, qc/, dev_calibration/, original_repro/
                     Data audit, split and leakage checks, reproduction of the earlier
                     protocol, pipeline verification, development-wave calibration
  hpo_results*.csv, selected_configs*.json
                     Hyper-parameter searches and selected configurations
```

## Requirements

- Python 3.13
- The packages pinned in `code/requirements.txt`
- A CPU is sufficient; no GPU is needed.

```bash
pip install -r code/requirements.txt
```

## Quick verification (no training, a few minutes)

Run from the repository root:

```bash
python3 code/scripts/00_sanity_tests.py          # unit checks of the agents
python3 code/scripts/02_preprocessing_audit.py   # rebuilds data/processed and the leakage checks
python3 code/scripts/22_reporting_summaries.py   # summaries from the saved result tables
```

## Full reproduction

```bash
bash code/run_all.sh --fresh --workers 4   # about 2.5-3.5 h on 4 CPU cores
```

`--fresh` deletes every generated result and rebuilds everything from the raw XML files, in this order:

1. Sanity tests of the agents (`00`).
2. Provenance audit and reconstruction of the earlier, leaky protocol (`01`, `01b`).
3. Preprocessing, splits and leakage assertions (`02`); development-wave calibration (`dev_calibration.py`).
4. Hyper-parameter search, selection, final evaluation, ablation and reward-ratio runs (`03`).
5. Aggregation, statistics and SHAP attributions (`04`, `07`); result tables (`05`).
6. Post-hoc baselines (`11`), search-draw sensitivity (`13`), three search draws for every model family (`19`), and exposure-only searches, last-epoch retraining and event-level metrics (`20`).
7. Score-scale, pipeline-verification, threshold-scale and reporting checks (`14`, `15`, `16`, `21`, `22`).

The trained-model files (`results/runs/`, about 300 MB) are not included; `run_all.sh` regenerates them. Steps `04`, `14`, `15` and `16` read these files, so they require a full run first. The scripts also write figures, LaTeX tables and logs, which are not part of this repository (see `.gitignore`).

**Random seeds.** Data splitting: 20261012 (outer) and 20261013-20261017 (inner). Hyper-parameter search: 777, with 801-802 for the post-hoc baselines and 21 configuration seeds between 9120 and 9208 for the further search draws (set in `code/configs/posthoc.json`). Final training: 1000-1004. Perturbations: 4242.

**Determinism.** Every process runs on the CPU with one thread, so results are deterministic on a given platform; minor floating-point differences across hardware are possible.

## Evaluation design

- **Data:** the 35-session curated UFFCSData release is used for evaluation; the 37 October-2019 sessions are used only for development.
- **Unit of independence:** session-linkage groups reconstructed from questionnaire profiles and session timing, because the release contains no participant identifiers. The checks establish session- and group-disjoint evaluation, not participant independence.
- **Splits:** nested grouped cross-validation (5 outer × 3 inner folds).
- **Features:** causal features only; all learned preprocessing is fitted on training groups.
- **Comparison:** every model receives the same search procedure and budget, checkpoint rule, cross-fitting recipe, operating rules and paired statistics.
- **Primary decision rule:** the threshold that minimizes the 5:1 cost on validation predictions.
- **Inference:** group bootstrap (primary) and exact group sign-flip test (sensitivity), with Holm adjustment.

The evaluation protocol and the post-hoc specifications were committed, with timestamps, to the authors' development repository before the corresponding runs. This repository is a curated copy of that work.

## Data

UFFCSData was collected and released by Porcino et al. (<https://github.com/tmp1986/UFFCSData>) under the GNU AGPL-3.0; its licence is included in `data/raw/UFFCSData-master_raw_source/LICENSE`. This repository includes:

- the curated release (`DATABASE_SELECTED_DATA`);
- the raw exports (`RAW_DATABASE`);
- `dataset.csv`, used only to reconstruct the earlier protocol.

The game build and the Weka files of the original release are not included; they are available from the original repository.

## Citation

The manuscript is currently under review at *PeerJ Computer Science*. If you use this code or these results, please cite the article once it is published; the full citation and DOI will be added here.

## Licence

The code is released under the MIT License (see `LICENSE`). The included UFFCSData files remain under the GNU AGPL-3.0.

## Contact

Rajai Darabee, Faculty of Science, Management & Computing, Universiti Teknologi PETRONAS, Malaysia (rajai_24006920@utp.edu.my).
