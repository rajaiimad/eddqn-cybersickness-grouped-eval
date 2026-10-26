"""Executable checks of the corrected learning procedure (Phase 3).

Run: python scripts/00_sanity_tests.py   (exits non-zero on any failure)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eddqn.agents import (ASYMMETRIC, AgentConfig, Replay, ValueAgent, _sessions_to_stream,  # noqa: E402
                          implied_probability, reward_table)
from eddqn.metrics import BAYES_THRESHOLD, binary_metrics, mcnemar, threshold_min_cost  # noqa: E402
from eddqn.supervised import unweight  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# 1. reward table implements the 5:1 FN:FP regret ratio
rt = reward_table("asymmetric")
regret_fn = rt["tp"] - rt["fn"]   # cost of missing an event (vs. alarming)
regret_fp = rt["tn"] - rt["fp"]   # cost of a false alarm (vs. staying silent)
check("asymmetric reward: FN regret / FP regret == 5", abs(regret_fn / regret_fp - 5) < 1e-12,
      f"{regret_fn:.2f}/{regret_fp:.2f}")
check("implied Bayes threshold == 1/6", abs(regret_fp / (regret_fn + regret_fp) - BAYES_THRESHOLD) < 1e-12)
for r in (1, 2, 5, 10):
    t = reward_table(f"ratio:{r}")
    check(f"ratio:{r} regret ratio", abs((t["tp"] - t["fn"]) / (t["tn"] - t["fp"]) - r) < 1e-12)
agent = ValueAgent(3, AgentConfig(seed=0, hidden=(8,), norm="none", warmup=0))
a = np.array([1, 1, 0, 0])
y = np.array([1, 0, 1, 0])
check("reward assignment TP/FP/FN/TN", np.allclose(agent._r(a, y), [0.5, -0.1, -0.5, 0.1]))

# 2. implied probability maps the myopic optimum back to p
p = np.linspace(0, 1, 11)
q1 = p * rt["tp"] + (1 - p) * rt["fp"]
q0 = p * rt["fn"] + (1 - p) * rt["tn"]
check("implied_probability inverts E[r|s,a]", np.allclose(implied_probability(np.c_[q0, q1], rt), p))
check("argmax Q  <=>  p >= 1/6", np.array_equal(q1 >= q0, p >= BAYES_THRESHOLD - 1e-12))
check("unweight(0.5) == 1/6 for w=5", abs(unweight(np.array([0.5]))[0] - 1 / 6) < 1e-6)

# 3. transitions never cross sessions; terminal flag at each session end
X = np.arange(20, dtype=np.float32).reshape(10, 2)
sess = {"A": np.array([0, 1, 2, 3]), "B": np.array([4, 5, 6]), "C": np.array([7, 8, 9])}
si, ni, dn = _sessions_to_stream(X, None, sess, ["B", "A", "C"])
owner = {i: k for k, v in sess.items() for i in v}
check("next state is in the same session", all(owner[s] == owner[n] for s, n in zip(si, ni)))
check("next state is the next second", all((n == s + 1) or d == 1 for s, n, d in zip(si, ni, dn)))
check("exactly one terminal per session at its last sample",
      dn.sum() == 3 and all(dn[list(si).index(v[-1])] == 1 for v in sess.values()))

# 4. no teacher forcing: stored action/reward pairs are consistent; full-info
#    replay stores the counterfactual action with ITS OWN reward
for full in (False, True):
    ag = ValueAgent(2, AgentConfig(seed=1, hidden=(8,), norm="none", warmup=0, full_info=full))
    buf = Replay(100, 2)
    acts = np.array([0, 1, 0, 1])
    labs = np.array([1, 1, 0, 0])
    ag._store(buf, X, np.array([0, 1, 2, 3]), np.array([1, 2, 3, 3]), np.array([0, 0, 0, 1.0]), acts, labs)
    ok = np.allclose(buf.r[:buf.n], ag._r(buf.a[:buf.n], buf.y[:buf.n]))
    check(f"replay reward == r(stored action, label) (full_info={full})", ok, f"n={buf.n}")

# 5. Polyak update also synchronises BatchNorm running statistics
ag = ValueAgent(4, AgentConfig(seed=2, hidden=(16,), norm="batch", tau=0.5))
ag.online.train()
ag.online(torch.randn(64, 4) * 5 + 3)       # moves online BN statistics
ag._sync_target()
bn_o = [b for b in ag.online.buffers()]
bn_t = [b for b in ag.target.buffers()]
check("target BN buffers == online BN buffers after soft update",
      all(torch.equal(a_, b_) for a_, b_ in zip(bn_o, bn_t)))

# 6. learning sanity on a separable toy problem (no collapse, sensible scores)
rng = np.random.default_rng(0)
n = 1200
Xt = rng.normal(size=(n, 3)).astype(np.float32)
yt = (Xt[:, 0] > 0.3).astype(int)
sid = np.repeat(np.arange(12), n // 12)
for gamma in (0.0, 0.9):
    ag = ValueAgent(3, AgentConfig(seed=3, hidden=(32, 32), norm="layer", dropout=0.0, gamma=gamma,
                                   epochs=15, eval_every=5, warmup=500, batch_size=128, learn_every=1,
                                   updates_per_learn=2, chunk=64))
    ag.fit(Xt, yt, sid, Xt, yt)
    s = ag.score(Xt)
    m = binary_metrics((s >= 1 / 6).astype(int), yt)
    q = ag.q_values(Xt)
    check(f"toy problem learnt (gamma={gamma}): balanced acc > 0.9", m["balanced_accuracy"] > 0.9,
          f"bal_acc={m['balanced_accuracy']:.3f}, |Q|max={np.abs(q).max():.2f}")
    bound = max(abs(v) for v in ASYMMETRIC.values()) / (1 - gamma) + 1e-6
    check(f"Q-values within the return bound (gamma={gamma})", np.abs(q).max() <= bound * 1.5,
          f"bound={bound:.2f}")

# 7. metrics / statistics
yy = np.array([1, 1, 1, 0, 0, 0, 1, 0])
pa = np.array([1, 1, 0, 0, 0, 1, 1, 0])
pb = np.array([1, 0, 0, 1, 0, 0, 1, 1])
mc = mcnemar(pa, pb, yy)
check("McNemar: b - c == errors(B) - errors(A)", mc["b_A_right_B_wrong"] - mc["c_A_wrong_B_right"] == mc["errors_B"] - mc["errors_A"])
sc = np.array([0.9, 0.8, 0.1, 0.05, 0.2, 0.3, 0.7, 0.01])
t = threshold_min_cost(sc, yy)
check("min-cost threshold computed on given data only", 0 <= t <= 1, f"t={t:.3f}")

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
sys.exit(1 if fails else 0)
