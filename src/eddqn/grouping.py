"""Conservative session-linkage groups used as the independent splitting unit.

The UFFCSData release contains **no participant identifier**.  A session is
one ~5-min play-through of one game.  Two facts show that session != person:

* aborted sessions are followed within minutes by a new session with the
  same questionnaire profile (e.g. a 104-s session ending 21:00:39 followed by
  a 302-s session at 21:02:27 with an identical 9-field profile), i.e. the
  same participant restarting; and
* several full-length sessions share an identical profile.

We therefore never treat sessions as independent when they *could* belong to
the same person.  Sessions are linked (union-find) when

  (a) their 9-field CSPQ profile is identical, or
  (b) the earlier one was aborted (< ``ABORT_ROWS`` samples), the later one
      starts within ``RESTART_GAP_S`` seconds of its end, and the profiles
      differ in at most ``MAX_PROFILE_DIFF`` fields.

Connected components are "linkage groups".  This over-merges (two different
people with the same profile end up in one group) which only makes the
evaluation more conservative; it can never place one person's sessions on
both sides of a split *unless* that person answered the questionnaire
differently in two non-consecutive sessions, which the data cannot reveal.
These groups are NOT participant identifiers and are never used as features.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from .io_raw import PROFILE

ABORT_ROWS = 200
RESTART_GAP_S = 180
MAX_PROFILE_DIFF = 2


def session_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in long.groupby("session_id", sort=False):
        first = g.iloc[0]
        start = first["start_time"]
        rows.append({
            "session_id": sid, "game": first["game"], "wave": first["wave"],
            "start_time": start, "end_time": start + _dt.timedelta(seconds=float(g["TimeStamp"].max())),
            "n_rows": len(g), "profile": tuple(first[c] for c in PROFILE),
            "n_pos": int((g["DiscomfortLevel"] >= 1).sum()),
        })
    return pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)


def linkage_groups(sessions: pd.DataFrame) -> pd.DataFrame:
    parent = {s: s for s in sessions.session_id}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, why, log):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
        log.append((a, b, why))

    log: list[tuple[str, str, str]] = []
    recs = sessions.to_dict("records")
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            if a["profile"] == b["profile"]:
                union(a["session_id"], b["session_id"], "identical profile", log)
    for a, b in zip(recs, recs[1:]):  # chronologically consecutive sessions
        gap = (b["start_time"] - a["end_time"]).total_seconds()
        diff = sum(x != y for x, y in zip(a["profile"], b["profile"]))
        if a["n_rows"] < ABORT_ROWS and 0 <= gap <= RESTART_GAP_S and 0 < diff <= MAX_PROFILE_DIFF:
            union(a["session_id"], b["session_id"],
                  f"restart after aborted session (gap {gap:.0f}s, {diff} profile field(s) differ)", log)
    roots = {s: find(s) for s in sessions.session_id}
    order = {r: i + 1 for i, r in enumerate(dict.fromkeys(roots[s] for s in sessions.session_id))}
    out = sessions.copy()
    out["group_id"] = [f"G{order[roots[s]]:02d}" for s in out.session_id]
    out.attrs["link_log"] = log
    return out
