# E-DDQN cybersickness detection: leakage-controlled, session-linkage-grouped evaluation

Code, data and results for the paper:

> **Does an Asymmetric-Reward E-DDQN Improve Safety-Prioritized Cybersickness Detection in Virtual Reality? A Leakage-Controlled Comparison with Standard DQN and Cost-Sensitive Baselines**
> R. Darabee, M. Muniandy, O. Shindi, C. W. Khuen, E. H. Sumiea.

This repository regenerates every experimental result of the paper from the raw UFFCSData session files. It also contains the saved per-sample predictions of every model, so the results can be checked without retraining.

## Contents

| Path | Content |
|---|---|
| `src/eddqn/` | Raw-data parser, session-linkage grouping, causal features, E-DDQN and baseline models (standard DQN, E-CB, CS-MLP, logistic regression, HGB, random forest), metrics, statistics and telemetry perturbations |
| `scripts/` | The pipeline steps, run in order by `run_all.sh` (see below) |
| `configs/protocol.json` | The evaluation protocol, fixed before any model was trained on the evaluation sessions |
| `configs/posthoc.json` | Specification of the post-hoc analyses, stating which were committed before they were run and which were added later |
| `configs/attacks.json` | Telemetry-perturbation scenarios |
| `data/processed/` | Outer and inner split tables (session-linkage groups) and perturbation masks |
| `results/predictions/` | Per-sample predictions of every model, seed and outer fold (`*.csv.gz`) |
| `results/tables/` | Metric tables, confidence intervals, paired tests, ablation, reward ratio, perturbations, post-hoc and search-draw analyses; `latex/` and `latex_supplement/` hold the same numbers as LaTeX tables |
| `results/figures/` | ROC/PR curves, confusion matrices, cost curves, SHAP |
| `results/preprocessing/`, `results/audit/`, `results/qc/` | Data audit, split and leakage checks, reproduction of the earlier protocol, pipeline verification |
| `results/hpo_results*.csv`, `results/selected_configs*.json` | Hyper-parameter search results and the selected configurations |
| `UFFCSData-master_raw_source/` | The UFFCSData files needed by the pipeline (see *Data*) |
| `DRL_cybersickness_FAST_v2.py` | The earlier, unpublished E-DDQN implementation, run unmodified to reconstruct the oversampling-before-splitting protocol |

## Reproduce

Requirements: Python 3.13 and the packages pinned in `requirements.txt`. A CPU is sufficient.

```bash
pip install -r requirements.txt
bash run_all.sh --fresh --workers 4   # about 2.5-3.5 h on 4 CPU cores
```

`--fresh` deletes every generated result and rebuilds everything from the raw XML files:

1. Sanity tests of the agents (`00`).
2. Provenance audit and reconstruction of the earlier, leaky protocol (`01`, `01b`).
3. Preprocessing, splits and leakage assertions (`02`), development-wave calibration (`dev_calibration.py`).
4. Hyper-parameter search, selection, final evaluation, ablation and reward-ratio runs (`03`).
5. Aggregation, statistics, figures and SHAP (`04`, `07`), result tables (`05`).
6. Post-hoc baselines and analyses (`11`), search-draw sensitivity (`13`), three search draws for every model family with draw-averaged comparisons (`19`), and exposure-only searches for every family, last-epoch retraining and event-level metrics (`20`).
7. Score-scale, pipeline-verification and threshold-scale checks (`14`, `15`, `16`, `21`).

The trained-model files (`results/runs/`, about 300 MB) are not included; `run_all.sh` regenerates them. Steps `04`, `14`, `15` and `16` read them, so they need a full run first.

Seeds: data splitting 20261012 (outer) and 20261013–20261017 (inner); hyper-parameter search 777 (801–802 for the post-hoc baselines; six configuration seeds between 9120 and 9202 for the search-draw analysis); final training 1000–1004; perturbations 4242. Every process runs on CPU with one thread, so results are deterministic on a given platform; minor floating-point differences across hardware are possible.

## Key design decisions

* **Data:** the 35-session curated UFFCSData release is used for evaluation; the 37 October-2019 sessions are used only for development.
* **Unit of independence:** session-linkage groups reconstructed from questionnaire profiles and session timing, because the release has no participant identifiers. The checks establish session- and group-disjoint evaluation, not participant independence.
* **Splits:** nested grouped cross-validation (5 outer × 3 inner folds).
* **Features:** causal features only; all learned preprocessing is fitted on training groups.
* **Fair comparison:** every model gets the same search procedure and budget, checkpoint rule, cross-fitting recipe, operating rules and paired statistics.
* **Primary decision rule:** the threshold minimising the 5:1 cost on validation predictions.
* **Inference:** group bootstrap (primary) and exact group sign-flip test (sensitivity), with Holm adjustment.

The protocol and the post-hoc specifications were committed, with timestamps, in the authors' development repository before the corresponding runs; this repository is a curated copy and starts with a single snapshot.

## Data

UFFCSData was collected and released by Porcino et al. (<https://github.com/tmp1986/UFFCSData>) under the GNU AGPL-3.0; its licence is in `UFFCSData-master_raw_source/LICENSE`. Included here:

* the curated release (`DATABASE_SELECTED_DATA`);
* the raw exports (`RAW_DATABASE`);
* `dataset.csv`, used only for the reconstruction of the earlier protocol.

The game build and the Weka files of the original release are not included; they are available from the original repository.

## Licence

Code: to be chosen by the authors. The included UFFCSData files remain under AGPL-3.0.

## Cite

Please cite the paper (citation to be added on publication).
