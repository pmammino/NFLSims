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

import numpy as np

import field_builder as fb

# ---- realistic-field defaults (ported from DFSSimsFull/app.py) -------------
# A pure ownership-imitation field is too soft: it hands a no-skill random
# lineup a positive edge, which is impossible. Real entrants build stacked,
# ceiling-seeking lineups, and the field you actually face is the BETTER of the
# lineups thousands of people submit — not raw builder draws. Two knobs fix it:
#   1. SHARP vs chalk: FIELD_SHARP_FRAC of the field is built "sharp" (bigger QB
#      stacks, ceiling-weighted skill players, more bring-backs) on the ownership
#      base so it still stacks popular teams; the rest is chalk (a soft tail).
#   2. SUBMITTED, not random: overbuild FIELD_OVERBUILD x and keep the highest-
#      projection contest_size lineups, so the field body is the submitted set.
FIELD_SHARP_FRAC = 0.75        # share built sharp vs chalk
FIELD_OVERBUILD = 2.0          # build this x contest size, keep top by projection
FIELD_OVERBUILD_CAP = 12000    # cap on EXTRA lineups built (perf guard)
FIELD_SHARP_STACK_STRENGTH = 0.5   # candidate_stack_sizes tilt for the sharp part
FIELD_SHARP_BRINGBACK = 0.35   # sharp lineups' forced primary-opponent bring-back


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


def _build_n(builder, n, cap_mult=40):
    """Draw `n` valid lineups from `builder` (bounded retries)."""
    lus, fails, limit = [], 0, n * cap_mult + 1000
    while len(lus) < n and fails < limit:
        lu = builder.build_one()
        if lu is None:
            fails += 1
            continue
        lus.append(lu)
    return lus


def _lineup_proj(lu, dk_mean):
    """A lineup's projection = sum of its players' mean sim points."""
    return float(sum(dk_mean.get(pl["key"], 0.0) for pl in lu["players"]))


def build_field(entities, params, n, dk_mean=None, *, n_med=6000,
                chalk_sensitivity=0.30, stack_tilt=0.12,
                sharp_frac=FIELD_SHARP_FRAC, overbuild=FIELD_OVERBUILD,
                overbuild_cap=FIELD_OVERBUILD_CAP,
                sharp_stack_strength=FIELD_SHARP_STACK_STRENGTH,
                sharp_bringback=FIELD_SHARP_BRINGBACK, seed=101,
                own_uncertainty=False, own_batch=200, own_corr=None):
    """Build a REALISTIC opponent field of `n` lineups.

    Combines three realism steps over the naive ownership-imitation field:
      * size model — chalk temperature (beta) + stack-shape tilt for the field
        size (via ``prepare_field_pool``);
      * sharp/chalk mix — ``sharp_frac`` of the build is ceiling-weighted with
        bigger stacks and forced bring-backs (still ownership-anchored so it
        stacks popular teams), the rest is plain chalk;
      * submitted-not-random — overbuild ``overbuild x n`` and keep the top `n`
        by projection (needs ``dk_mean`` {key: mean sim points}; without it the
        overbuild is skipped and all built lineups are kept in draw order).

    ``own_uncertainty`` rebuilds the (size-adjusted) pool from a fresh ownership
    draw every ``own_batch`` lineups so the field mixes over ownership scenarios.
    Returns a list of `n` lineup dicts. All knobs default to OFF/neutral values
    matching the field-imitation behaviour when overbuild=1 and sharp_frac=0."""
    n = int(n)
    extra = 0
    if dk_mean is not None and overbuild > 1.0:
        extra = min(int(round(n * (overbuild - 1.0))), int(overbuild_cap))
    total = n + extra
    n_sharp = int(round(sharp_frac * total))

    def _pool_params():
        adj, p_sz, beta = prepare_field_pool(
            entities, params, n, n_med=n_med,
            chalk_sensitivity=chalk_sensitivity, stack_tilt=stack_tilt)
        return adj, p_sz, beta

    adj, p_sz, beta = _pool_params()
    p_sharp = dict(p_sz)
    p_sharp["stack_sizes"] = fb.candidate_stack_sizes(
        p_sz["stack_sizes"], sharp_stack_strength)

    unc_rng = np.random.default_rng(seed + 7) if own_uncertainty else None

    def _mk(sharp, built_so_far):
        pool_entities = adj
        if unc_rng is not None:
            from ownership_model import resample_ownership
            drawn = resample_ownership(
                entities, unc_rng, corr=(own_corr if own_corr is not None else 0.35))
            pool_entities = adjust_ownership(
                normalize_to_slots(drawn, pos_slot_targets(params)),
                beta_for_size(n, n_med, chalk_sensitivity))
        p = p_sharp if sharp else p_sz
        return fb.Builder(fb.Pool(pool_entities), p, seed=seed + n + built_so_far,
                          uniform=False,
                          use_upside=sharp,
                          bringback_prob=(sharp_bringback if sharp else None))

    lineups = []
    for sharp, count in ((True, n_sharp), (False, total - n_sharp)):
        remaining = count
        while remaining > 0:
            batch = remaining if unc_rng is None else min(remaining, int(own_batch))
            b = _mk(sharp, len(lineups))
            got = _build_n(b, batch)
            lineups.extend(got)
            remaining -= batch
            if not got and unc_rng is None:
                break

    if dk_mean is not None and len(lineups) > n:
        lineups.sort(key=lambda lu: _lineup_proj(lu, dk_mean), reverse=True)
        lineups = lineups[:n]
    return lineups


if __name__ == "__main__":
    import nfl_ingest
    slate = nfl_ingest.build_slate()
    params = fb.load_params()
    for N in (1000, 6000, 20000):
        adj, p, beta = prepare_field_pool(slate.entities, params, N)
        print(f"N={N:>6}  beta={beta:.3f}  "
              f"stack_sizes={[ (k, round(w,3)) for k,w in p['stack_sizes'] ]}")
