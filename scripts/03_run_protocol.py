"""Run the frozen nested protocol on the curated evaluation sessions.

Stages (each resumable; one pickle per job under results/runs/):
  hpo       random search per family, inner CV only (outer-test rows are not
            even featurised in this stage)
  select    pick the configuration per (family, outer fold) from pooled
            inner-OOF validation predictions -> results/selected_configs.json
  final     cross-fitted training with the selected configuration for every
            seed; scores the clean outer-test fold and every frozen
            perturbation variant (configs/attacks.json)
  ablation  leave-one-component-out variants of E-DDQN (selected E-DDQN
            configuration, final stage only, no re-tuning)
  ratio     reward-ratio sensitivity of E-DDQN (1:1, 2:1, 5:1, 10:1)

Usage: python scripts/03_run_protocol.py <stage> [families...] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from eddqn import CONFIGS, PROCESSED, RESULTS  # noqa: E402
from eddqn.attacks import load_attack_config, perturb_long, variants  # noqa: E402
from eddqn.data import build_datasets, fold_frames, prepare_fold, split_tables  # noqa: E402
from eddqn.experiment import AGENT_FIXED, run_job, sample_configs  # noqa: E402
from eddqn.metrics import min_cost_value, ranking_metrics  # noqa: E402

PROTOCOL = json.loads((CONFIGS / "protocol.json").read_text())
RUNS = RESULTS / "runs"
FAMILIES = ["E-DDQN", "E-CB", "DQN", "CS-MLP", "LogReg", "HGB", "RF"]
_CACHE = {}


def family_space(fam):
    sp = PROTOCOL["search_spaces"]
    if fam in AGENT_FIXED:
        space = {**sp["_neural_common"], **sp["_agent_common"], **sp[fam]}
        fixed = {**PROTOCOL["fixed_training"]["agents"], **AGENT_FIXED[fam]}
    elif fam == "CS-MLP":
        space = {**sp["_neural_common"], **sp["CS-MLP"]}
        fixed = dict(PROTOCOL["fixed_training"]["CS-MLP"])
    else:
        space, fixed = dict(sp[fam]), {}
    for k in list(space):          # keys fixed by the family definition are not searched
        if k in fixed:
            space.pop(k)
    return space, fixed


def configs_for(fam):
    space, fixed = family_space(fam)
    n = PROTOCOL["hpo_budget"][fam]
    seed = PROTOCOL["hpo_seed"] + FAMILIES.index(fam)
    return sample_configs(space, fixed, n, seed)


def _worker(job):
    path = Path(job["path"])
    if path.exists():
        return str(path)
    out = run_job(job, _CACHE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    tmp.rename(path)
    return str(path)


def run_jobs(jobs, workers):
    t0 = time.time()
    todo = [j for j in jobs if not Path(j["path"]).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run, {workers} workers", flush=True)
    if not todo:
        return
    ctx = get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, p in enumerate(pool.imap_unordered(_worker, todo, chunksize=1), 1):
            if i % max(1, len(todo) // 20) == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} done, {el/60:.1f} min elapsed, ETA {el/i*(len(todo)-i)/60:.1f} min", flush=True)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- stages
def stage_hpo(long, tables, fams, workers):
    _CACHE.clear()
    from eddqn.experiment import build_cache
    _CACHE.update(build_cache(long, tables, PROTOCOL, include_test=False))
    jobs = []
    for fam in fams:
        for ci, cfg in enumerate(configs_for(fam)):
            for k in range(PROTOCOL["k_outer"]):
                for j in range(PROTOCOL["k_inner"]):
                    jobs.append({"family": fam, "config": cfg, "config_id": ci, "outer": k, "inner": j,
                                 "seed": PROTOCOL["hpo_seed"], "stage": "hpo",
                                 "path": str(RUNS / "hpo" / fam / f"o{k}_c{ci:02d}_i{j}.pkl")})
    # longest families first for better load balance
    jobs.sort(key=lambda j: (j["family"] not in AGENT_FIXED, j["family"]))
    run_jobs(jobs, workers)


def stage_select(long, tables, fams):
    sel_path = RESULTS / "selected_configs.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    rows = []
    for fam in fams:
        cfgs = configs_for(fam)
        sel[fam] = {}
        for k in range(PROTOCOL["k_outer"]):
            best = None
            for ci, cfg in enumerate(cfgs):
                parts = [load(RUNS / "hpo" / fam / f"o{k}_c{ci:02d}_i{j}.pkl") for j in range(PROTOCOL["k_inner"])]
                ys = []
                for j in range(PROTOCOL["k_inner"]):
                    _, val, _ = fold_frames(long, tables, k, j)
                    ys.append(val["y"].to_numpy())
                s, y = np.concatenate([p["val_score"] for p in parts]), np.concatenate(ys)
                crit, prauc = min_cost_value(s, y), ranking_metrics(s, y)["pr_auc"]
                rows.append({"family": fam, "outer": k, "config_id": ci, "pooled_oof_min_cost": crit,
                             "pooled_oof_pr_auc": prauc, "pooled_oof_roc_auc": ranking_metrics(s, y)["roc_auc"],
                             "mean_seconds": float(np.mean([p["seconds"] for p in parts])),
                             **{f"cfg_{a}": (str(b) if isinstance(b, (list, tuple)) else b) for a, b in cfg.items()}})
                key = (crit, -prauc)
                if best is None or key < best[0]:
                    best = (key, ci, cfg)
            sel[fam][str(k)] = {"config_id": best[1], "config": best[2],
                                "pooled_oof_min_cost": best[0][0], "pooled_oof_pr_auc": -best[0][1]}
    sel_path.write_text(json.dumps(sel, indent=2, default=str))
    hpo = pd.DataFrame(rows)
    old = RESULTS / "hpo_results.csv"
    if old.exists():
        prev = pd.read_csv(old)
        hpo = pd.concat([prev[~prev.family.isin(fams)], hpo])
    hpo.to_csv(old, index=False)
    print(pd.DataFrame([{"family": f, "outer": k, **{a: b for a, b in v.items() if a != "config"},
                         "feature_set": v["config"].get("feature_set"), "gamma": v["config"].get("gamma")}
                        for f, d in sel.items() for k, v in d.items() if f in fams]).to_string())


def _final_cache(long, tables, with_attacks=True):
    _CACHE.clear()
    from eddqn.experiment import build_cache
    _CACHE.update(build_cache(long, tables, PROTOCOL, include_test=True))
    if not with_attacks:
        return
    acfg = load_attack_config()
    var = variants(acfg)
    zmask = {}
    for k in range(PROTOCOL["k_outer"]):
        _, test = fold_frames(long, tables, k)
        pert = {}
        for v in var:
            pl, z = perturb_long(test, v, acfg["seed"])
            pert[v["name"]] = pl
            zmask[(k, v["name"])] = z
        for j in range(PROTOCOL["k_inner"]):
            tr, val, _ = fold_frames(long, tables, k, j)
            fd = prepare_fold(tr, {name: pl for name, pl in pert.items()}, with_profile=True)
            _CACHE[("attacks", k, j)] = fd.X_ev
    _CACHE["zmask"] = zmask
    with open(PROCESSED / "attack_masks.pkl", "wb") as f:
        pickle.dump({"variants": var, "z": zmask}, f)


def final_jobs(fam, cfg_by_outer, tag=None, seeds=None, score_attacks=True):
    jobs = []
    for k in range(PROTOCOL["k_outer"]):
        cfg = cfg_by_outer[str(k)]["config"] if "config" in cfg_by_outer[str(k)] else cfg_by_outer[str(k)]
        for s in (seeds or PROTOCOL["final_seeds"]):
            for j in range(PROTOCOL["k_inner"]):
                name = tag or fam
                jobs.append({"family": fam, "config": cfg, "config_id": -1, "outer": k, "inner": j, "seed": 1000 + s,
                             "stage": "final", "score_attacks": score_attacks, "tag": name,
                             "path": str(RUNS / "final" / name / f"o{k}_s{s}_i{j}.pkl")})
    return jobs


def stage_final(long, tables, fams, workers):
    sel = json.loads((RESULTS / "selected_configs.json").read_text())
    _final_cache(long, tables, with_attacks=True)
    jobs = []
    for fam in fams:
        jobs += final_jobs(fam, sel[fam])
    jobs.sort(key=lambda j: (j["family"] not in AGENT_FIXED, j["family"]))
    run_jobs(jobs, workers)


def stage_ablation(long, tables, workers):
    sel = json.loads((RESULTS / "selected_configs.json").read_text())["E-DDQN"]
    _final_cache(long, tables, with_attacks=False)
    jobs = []
    for label, change in PROTOCOL["ablation_components"].items():
        cfgs = {}
        for k, v in sel.items():
            c = dict(v["config"])
            for a, b in change.items():
                c[a] = (not c.get(a, False)) if b == "toggle" else b
            cfgs[k] = {"config": c}
        tag = "ABL " + label.replace("/", "-").replace(" ", "_")
        jobs += final_jobs("E-DDQN", cfgs, tag=tag, score_attacks=False)
    run_jobs(jobs, workers)


def stage_ratio(long, tables, workers):
    sel = json.loads((RESULTS / "selected_configs.json").read_text())["E-DDQN"]
    _final_cache(long, tables, with_attacks=False)
    jobs = []
    for r in PROTOCOL["reward_ratio_sensitivity"]:
        cfgs = {k: {"config": {**v["config"], "reward": f"ratio:{r}"}} for k, v in sel.items()}
        jobs += final_jobs("E-DDQN", cfgs, tag=f"RATIO {r}", score_attacks=False)
    run_jobs(jobs, workers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("families", nargs="*", default=FAMILIES)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    long, _, _ = build_datasets()
    tables = split_tables(long, PROTOCOL)
    # the split tables written by 02_preprocessing_audit.py must match
    saved = pd.read_csv(PROCESSED / "outer_folds.csv")
    assert saved[["group_id", "fold"]].sort_values("group_id").values.tolist() == \
        tables["outer"][["group_id", "fold"]].sort_values("group_id").values.tolist(), "split drift"
    {"hpo": lambda: stage_hpo(long, tables, a.families, a.workers),
     "select": lambda: stage_select(long, tables, a.families),
     "final": lambda: stage_final(long, tables, a.families, a.workers),
     "ablation": lambda: stage_ablation(long, tables, a.workers),
     "ratio": lambda: stage_ratio(long, tables, a.workers)}[a.stage]()
