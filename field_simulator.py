#!/usr/bin/env python3
"""
field_simulator.py
==================
Contest-size model on top of ``field_builder`` — the NFL port of
``DFSSimsFull/field_simulator.py``. Two size effects reshape the projected
ownership (assumed to describe a MEDIUM contest) relative to a baseline:

1. OWNERSHIP TEMPERATURE (the chalk knob). Within each position group ownership
   is reshaped as ``own^beta`` and renormalized so the per-slot total is
   preserved. ``beta = 1 - k*log10(N/N_med)``: beta>1 concentrates chalk (small
   fields), beta<1 flattens it (large fields).
2. STACK-SHAPE TILT. Larger fields consolidate onto bigger QB stacks; the
   stack-size distribution is mildly tilted toward k>=2 for large fields.

`normalize_to_slots` rescales ownership so each position sums to its DK slot
count x100% — including the FLEX slot, split across RB/WR/TE by the field
params' FLEX distribution.
"""
import copy
import math
from collections import defaultdict


def pos_slot_targets(params):
    """Per-position ownership target (x100%) including the FLEX allocation."""
    base = {"QB": 1.0, "RB": 2.0, "WR": 3.0, "TE": 1.0, "DST": 1.0}
    for pos, share in params.get("flex_pos", []):
        base[pos] = base.get(pos, 0.0) + float(share)
    return {p: v * 100.0 for p, v in base.items()}


def _by_pos(entities):
    groups = defaultdict(list)
    for i, e in enumerate(entities):
        groups[e["pos"]].append(i)
    return groups


def normalize_to_slots(entities, targets):
    """Rescale each position group's ownership to sum to its slot target.

    Raw projected ownership is often over/under-subscribed per position; this
    makes the field-fill targets feasible while preserving each player's
    relative ownership within its position."""
    out = [dict(e) for e in entities]
    for e in out:
        e["own"] = max(float(e.get("own", 0.0)), 1e-3)
    for pos, idx in _by_pos(out).items():
        target = targets.get(pos, 100.0)
        s = sum(out[i]["own"] for i in idx)
        if s > 0:
            for i in idx:
                out[i]["own"] = out[i]["own"] / s * target
    return out


def beta_for_size(n, n_med, k):
    return 1.0 - k * math.log10(max(n, 1) / n_med)


def adjust_ownership(entities, beta):
    """Reshape ownership as ``own^beta`` renormalized within each position so the
    per-slot total is preserved (chalk concentrates for beta>1)."""
    out = [dict(e) for e in entities]
    for e in out:
        e["own"] = max(float(e.get("own", 0.0)), 1e-3)
    for pos, idx in _by_pos(out).items():
        grp = [out[i]["own"] for i in idx]
        tot = sum(grp)
        reshaped = [x ** beta for x in grp]
        rs = sum(reshaped)
        if rs > 0:
            for i, r in zip(idx, reshaped):
                out[i]["own"] = r / rs * tot
    return out


def tilt_stacks(stack_sizes, n, n_med, s):
    """Tilt the QB-stack-size distribution toward bigger stacks for large fields.

    stack_sizes: list of [k, weight]. Multi-catcher stacks (k>=2) are up-weighted
    for N>N_med and down-weighted for N<N_med; naked/one-off (k<=1) inversely."""
    factor = 1.0 + s * math.log10(max(n, 1) / n_med)
    out = []
    for k, w in stack_sizes:
        if k >= 2:
            w = w * factor
        elif k <= 1:
            w = w / factor
        out.append([k, max(w, 1e-9)])
    tot = sum(w for _, w in out)
    return [[k, w / tot] for k, w in out]


def size_adjusted_params(params, n, n_med, chalk_sensitivity, stack_tilt):
    """Return a copy of `params` with the stack-size distribution tilted for N."""
    p = copy.deepcopy(params)
    p["stack_sizes"] = tilt_stacks(params["stack_sizes"], n, n_med, stack_tilt)
    return p


def prepare_field_pool(entities, params, n, n_med=6000,
                       chalk_sensitivity=0.30, stack_tilt=0.12):
    """Full pipeline for one field size: normalize to slots, then apply the
    chalk temperature. Returns (adjusted_entities, size_adjusted_params, beta)."""
    targets = pos_slot_targets(params)
    norm = normalize_to_slots(entities, targets)
    beta = beta_for_size(n, n_med, chalk_sensitivity)
    adj = adjust_ownership(norm, beta)
    p = size_adjusted_params(params, n, n_med, chalk_sensitivity, stack_tilt)
    return adj, p, beta


if __name__ == "__main__":
    import nfl_ingest
    import field_builder as fb
    slate = nfl_ingest.build_slate()
    params = fb.load_params()
    for N in (1000, 6000, 20000):
        adj, p, beta = prepare_field_pool(slate.entities, params, N)
        print(f"N={N:>6}  beta={beta:.3f}  "
              f"stack_sizes={[ (k, round(w,3)) for k,w in p['stack_sizes'] ]}")
