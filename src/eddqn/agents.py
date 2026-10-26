"""Corrected value-based agents: E-DDQN (proposed) and the standard DQN baseline.

Formulation (derived in the paper, Section III)
-----------------------------------------------
* episode      : one recorded play-through (session), in temporal order.
* state  s_t   : causal feature vector of the session at second t.
* action a_t   : 0 = no alarm, 1 = alarm.
* reward       : r(a_t, y_t) from the 2x2 table (asymmetric 5:1 for E-DDQN).
* transition   : s_{t+1} = the NEXT SECOND OF THE SAME SESSION; the last
                 sample of a session is terminal (done = 1, no bootstrap).
                 No transition ever links two different sessions.
* The logged telemetry does not depend on the alarm (exogenous context), so
  P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_t).  Hence
      Q*(s,a) = E[r | s,a] + gamma * E[V*(s') | s]
  and the second term is action-independent: the optimal policy is the
  myopic (contextual-bandit) policy for every gamma.  gamma is kept as a
  hyper-parameter so that validation data, not assumption, decides whether
  bootstrapping helps; gamma = 0 is the contextual-bandit special case.

Defects of the original script that are fixed here
--------------------------------------------------
1. teacher forcing (reward of action a stored with action y) - removed.
2. random cross-sample transitions (s' = next row of a shuffled batch) -
   replaced by within-session transitions with terminal flags.
3. Polyak update ignored BatchNorm running statistics - buffers are now
   copied, and the target net is evaluated in eval mode.
4. Double-DQN action selection ran the online net in train mode (dropout on,
   BN statistics updated by s') - now computed in eval mode.
5. Test data are never seen by the agent; checkpoints are chosen on
   validation groups only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import min_cost_value, ranking_metrics

ASYMMETRIC = {"tp": 0.5, "tn": 0.1, "fn": -0.5, "fp": -0.1}   # manuscript Eq. (9)
SYMMETRIC = {"tp": 1.0, "tn": 1.0, "fn": -1.0, "fp": -1.0}


def reward_table(name: str) -> dict:
    """'asymmetric' (paper, FN:FP regret 5:1), 'symmetric' (+-1), or
    'ratio:R' (TP +0.5, FN -0.5, TN +0.5/R, FP -0.5/R -> regret ratio R:1)."""
    if name.startswith("ratio:"):
        r = float(name.split(":")[1])
        return {"tp": 0.5, "fn": -0.5, "tn": 0.5 / r, "fp": -0.5 / r}
    return {"asymmetric": ASYMMETRIC, "symmetric": SYMMETRIC}[name]


def implied_probability(q: np.ndarray, rt: dict) -> np.ndarray:
    """Map Q-values to the posterior implied by a myopic value function:
    Q(s,1)-Q(s,0) = p*(r_tp - r_fn) - (1-p)*(r_tn - r_fp)."""
    d1 = rt["tp"] - rt["fn"]
    d0 = rt["tn"] - rt["fp"]
    return (q[:, 1] - q[:, 0] + d0) / (d1 + d0)


class QNet(nn.Module):
    def __init__(self, n_in, hidden=(256, 512, 256), dropout=0.2, norm="batch", dueling=True, n_out=2):
        super().__init__()
        layers, d = [], n_in
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(d, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout > 0 and i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout))
            d = h
        self.trunk = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.val = nn.Linear(d, 1)
            self.adv = nn.Linear(d, n_out)
        else:
            self.head = nn.Linear(d, n_out)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.trunk(x)
        if self.dueling:
            a = self.adv(h)
            return self.val(h) + a - a.mean(dim=1, keepdim=True)
        return self.head(h)


@dataclass
class AgentConfig:
    # objective / RL structure
    reward: str = "asymmetric"
    gamma: float = 0.0
    double: bool = True
    dueling: bool = True
    target_update: str = "soft"     # 'soft' (Polyak tau) | 'hard' (copy every hard_every updates)
    tau: float = 0.01
    hard_every: int = 10
    replay: str = "balanced"        # 'balanced' (rho Normal / 1-rho Attack) | 'uniform'
    rho: float = 0.5
    full_info: bool = False         # also store the counterfactual action with ITS reward
    warmup: int = 4000              # random-action transitions before the first update
    # exploration
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.97          # multiplicative, per epoch
    # network / optimisation
    hidden: tuple = (256, 512, 256)
    dropout: float = 0.2
    norm: str = "batch"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    buffer: int = 50000
    epochs: int = 60
    chunk: int = 256
    learn_every: int = 4             # chunks between learning phases
    updates_per_learn: int = 2
    consolidation: int = 5           # extra updates at the end of every epoch
    grad_clip: float = 1.0
    eval_every: int = 2
    seed: int = 0
    extra: dict = field(default_factory=dict)


class Replay:
    def __init__(self, cap, dim):
        self.s = np.zeros((cap, dim), np.float32)
        self.s2 = np.zeros((cap, dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.y = np.zeros(cap, np.int64)
        self.cap, self.n, self.i = cap, 0, 0

    def add(self, s, a, r, s2, d, y):
        k = len(a)
        idx = (self.i + np.arange(k)) % self.cap
        self.s[idx], self.a[idx], self.r[idx] = s, a, r
        self.s2[idx], self.d[idx], self.y[idx] = s2, d, y
        self.i = (self.i + k) % self.cap
        self.n = min(self.n + k, self.cap)

    def sample(self, bs, rng, balanced, rho):
        if balanced:
            lab = self.y[: self.n]
            i0, i1 = np.flatnonzero(lab == 0), np.flatnonzero(lab == 1)
            if len(i0) and len(i1):
                n0 = int(round(bs * rho))
                idx = np.concatenate([rng.choice(i0, n0, replace=len(i0) < n0),
                                      rng.choice(i1, bs - n0, replace=len(i1) < bs - n0)])
                return idx
        return rng.choice(self.n, bs, replace=self.n < bs)


def _sessions_to_stream(X, y, sess_ids, order):
    """Concatenate sessions (each in temporal order) following ``order`` and
    return per-row (state idx, next-state idx, done)."""
    rows, nxt, done = [], [], []
    for sid in order:
        idx = sess_ids[sid]
        rows.append(idx)
        nx = np.empty_like(idx)
        nx[:-1] = idx[1:]
        nx[-1] = idx[-1]           # unused because done = 1
        nxt.append(nx)
        dn = np.zeros(len(idx), np.float32)
        dn[-1] = 1.0
        done.append(dn)
    return np.concatenate(rows), np.concatenate(nxt), np.concatenate(done)


class ValueAgent:
    def __init__(self, n_in: int, cfg: AgentConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.rt = reward_table(cfg.reward)
        self.online = QNet(n_in, cfg.hidden, cfg.dropout, cfg.norm, cfg.dueling)
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(cfg.epochs, 1), eta_min=1e-5)
        self.n_updates = 0
        self.history = []

    # -------------------------------------------------------------- utils
    def _r(self, a, y):
        rt = self.rt
        return np.where(a == 1, np.where(y == 1, rt["tp"], rt["fp"]),
                        np.where(y == 1, rt["fn"], rt["tn"])).astype(np.float32)

    @torch.no_grad()
    def q_values(self, X, bs=4096):
        self.online.eval()
        out = [self.online(torch.from_numpy(X[i:i + bs])).numpy() for i in range(0, len(X), bs)]
        self.online.train()
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)

    def score(self, X):
        return implied_probability(self.q_values(X), self.rt)

    def _sync_target(self, hard=False):
        with torch.no_grad():
            if hard or self.cfg.target_update == "hard":
                self.target.load_state_dict(self.online.state_dict())
                return
            tau = self.cfg.tau
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.mul_(1 - tau).add_(tau * op)
            for tb, ob in zip(self.target.buffers(), self.online.buffers()):
                tb.copy_(ob)   # BatchNorm running statistics (fix of defect 3)

    def _update(self, buf):
        c = self.cfg
        idx = buf.sample(c.batch_size, self.rng, c.replay == "balanced", c.rho)
        s = torch.from_numpy(buf.s[idx])
        a = torch.from_numpy(buf.a[idx]).unsqueeze(1)
        r = torch.from_numpy(buf.r[idx]).unsqueeze(1)
        s2 = torch.from_numpy(buf.s2[idx])
        d = torch.from_numpy(buf.d[idx]).unsqueeze(1)
        self.online.train()
        q = self.online(s).gather(1, a)
        with torch.no_grad():
            if c.gamma > 0:
                if c.double:
                    self.online.eval()
                    a2 = self.online(s2).argmax(1, keepdim=True)
                    self.online.train()
                    q2 = self.target(s2).gather(1, a2)
                else:
                    q2 = self.target(s2).max(1, keepdim=True).values
                tgt = r + c.gamma * (1 - d) * q2
            else:
                tgt = r
        loss = F.smooth_l1_loss(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.opt.step()
        self.n_updates += 1
        if c.target_update == "soft":
            self._sync_target()
        elif self.n_updates % c.hard_every == 0:
            self._sync_target(hard=True)
        return float(loss.item())

    def _store(self, buf, X, si, ni, dn, a, y):
        r = self._r(a, y)
        buf.add(X[si], a, r, X[ni], dn, y)
        if self.cfg.full_info:   # counterfactual action with its own (correct) reward
            a2 = 1 - a
            buf.add(X[si], a2, self._r(a2, y), X[ni], dn, y)
        return r

    # ----------------------------------------------------------- training
    def fit(self, X, y, sessions, X_val=None, y_val=None, log=None):
        """``sessions``: array of session ids aligned with rows of X (rows of a
        session must be in temporal order)."""
        c = self.cfg
        sess_ids = {}
        for i, s in enumerate(sessions):
            sess_ids.setdefault(s, []).append(i)
        sess_ids = {k: np.asarray(v) for k, v in sess_ids.items()}
        keys = list(sess_ids)
        buf = Replay(c.buffer * (2 if c.full_info else 1), X.shape[1])

        # warm-up with uniformly random actions (valid transitions, Eq. (9) reward)
        if c.warmup > 0:
            si, ni, dn = _sessions_to_stream(X, y, sess_ids, self.rng.permutation(keys))
            k = min(c.warmup, len(si))
            a = self.rng.integers(0, 2, k)
            self._store(buf, X, si[:k], ni[:k], dn[:k], a, y[si[:k]])

        eps = c.eps_start
        best = (np.inf, -np.inf)
        best_state, best_epoch = copy.deepcopy(self.online.state_dict()), 0
        for ep in range(1, c.epochs + 1):
            si, ni, dn = _sessions_to_stream(X, y, sess_ids, self.rng.permutation(keys))
            greedy_reward, n_greedy = 0.0, 0
            for k, st in enumerate(range(0, len(si), c.chunk)):
                sl = slice(st, st + c.chunk)
                s_idx = si[sl]
                rand = self.rng.random(len(s_idx)) < eps
                a = self.rng.integers(0, 2, len(s_idx))
                if (~rand).any():
                    qv = self.q_values(X[s_idx[~rand]])
                    a[~rand] = qv.argmax(1)
                r = self._store(buf, X, s_idx, ni[sl], dn[sl], a, y[s_idx])
                greedy_reward += float(r[~rand].sum())
                n_greedy += int((~rand).sum())
                if (k + 1) % c.learn_every == 0 and buf.n >= c.batch_size:
                    for _ in range(c.updates_per_learn):
                        self._update(buf)
            if buf.n >= c.batch_size:
                for _ in range(c.consolidation):
                    self._update(buf)
            eps = max(c.eps_end, eps * c.eps_decay)
            self.sched.step()
            rec = {"epoch": ep, "eps": eps, "greedy_reward_per_step": greedy_reward / max(n_greedy, 1),
                   "updates": self.n_updates}
            if X_val is not None and (ep % c.eval_every == 0 or ep == c.epochs):
                sv = self.score(X_val)
                crit = min_cost_value(sv, y_val)
                prauc = ranking_metrics(sv, y_val)["pr_auc"]
                rec.update(val_min_cost=crit, val_pr_auc=prauc)
                key = (-crit, prauc)
                if key > (-best[0], best[1]):
                    best, best_epoch = (crit, prauc), ep
                    best_state = copy.deepcopy(self.online.state_dict())
            self.history.append(rec)
            if log:
                log(rec)
        if X_val is not None:
            self.online.load_state_dict(best_state)
        self.best_epoch = best_epoch if X_val is not None else c.epochs
        return self
