#!/usr/bin/env python3
"""
contest_sim.py
==============
Score lineups across the simulation and run them against a simulated field —
the NFL port of the ``score_matrix`` / ``run_contest`` core in
``DFSSimsFull/stage_d.py``.

Both operate on the stage-boundary artifact: ``dk`` = {entity_key:
np.ndarray[n_sims]} of DraftKings points. A lineup's per-sim total is the sum of
its 9 players' arrays; a candidate's finish in a sim is its rank within the
field's sorted totals.
"""
import numpy as np


def score_matrix(lineups, dk, n_sim):
    """(n_sim, n_lineups) per-sim DK totals. `lineups` are builder dicts with a
    'players' list of entity dicts carrying 'key'."""
    cols = []
    for lu in lineups:
        t = np.zeros(n_sim, dtype=np.float32)
        for pl in lu["players"]:
            arr = dk.get(pl["key"])
            if arr is not None:
                t += arr
        cols.append(t)
    return np.column_stack(cols) if cols else np.zeros((n_sim, 0), np.float32)


def run_contest(field_mat, cand_mat, n_sim, n_field):
    """For each sim, place every candidate against the sorted field.

    Returns (wins, top10, top100, avg_place). place = n_field - (#field strictly
    below the candidate) + 1, i.e. 1 = first."""
    n = cand_mat.shape[1]
    wins = np.zeros(n, np.int64)
    t10 = np.zeros(n, np.int64)
    t100 = np.zeros(n, np.int64)
    psum = np.zeros(n, np.int64)
    for s in range(n_sim):
        fs = np.sort(field_mat[s])
        cv = cand_mat[s]
        place = (n_field - np.searchsorted(fs, cv, side="right")) + 1
        wins += (place == 1)
        t10 += (place <= 10)
        t100 += (place <= 100)
        psum += place
    return wins, t10, t100, psum / n_sim


if __name__ == "__main__":
    # tiny hand-checkable case: 3-entry field, 2 candidates, 2 sims
    dk = {
        "A": np.array([10, 1], np.float32), "B": np.array([2, 2], np.float32),
        "C": np.array([3, 9], np.float32), "D": np.array([4, 4], np.float32),
    }
    field = [{"players": [{"key": "A"}]}, {"players": [{"key": "B"}]},
             {"players": [{"key": "C"}]}]
    cand = [{"players": [{"key": "D"}]}]
    fm = score_matrix(field, dk, 2)          # (2,3)
    cm = score_matrix(cand, dk, 2)           # (2,1)
    wins, t10, t100, avg = run_contest(fm, cm, 2, 3)
    # sim0 field sorted [2,3,10], D=4 -> beats 2,3 -> place 3-2+1=2
    # sim1 field sorted [1,2,9], D=4 -> beats 1,2 -> place 3-2+1=2
    assert list(avg) == [2.0], avg
    assert wins[0] == 0, wins
    print("contest_sim.py self-test passed")
