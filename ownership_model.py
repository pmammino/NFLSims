#!/usr/bin/env python3
"""
ownership_model.py
==================
NFL port of the ownership-uncertainty piece of ``DFSSimsFull/ownership_model.py``.

Projected ownership is a point estimate, but the real field is drawn from a
DISTRIBUTION around it — a chalky QB might come back 22% or 38%. Grading every
candidate against one fixed ownership vector treats %Rostered as a fact and
understates how much a candidate's finish depends on which ownership scenario
actually plays out.

``resample_ownership`` draws ONE ownership realization: each entity's ownership
is perturbed by a lognormal shock (proportional noise keeps it positive and
scales with the projection), optionally sharing a per-slate chalk shock so the
whole board leans chalky or contrarian together. Feeding a fresh draw into the
field build every N lineups makes the simulated field a MIXTURE over ownership
scenarios instead of a single point estimate.

The per-position renormalization the field build already does
(``field_simulator.normalize_to_slots``) turns any draw back into a valid field
composition, so this module only has to perturb the raw ``own`` values.
"""
import numpy as np

# proportional ownership noise (lognormal sigma) by position — passing-game
# skill spots (WR/TE) swing more slate-to-slate than the near-locked QB/DST or
# workhorse RB. Deliberately moderate; this is uncertainty, not chaos.
DEFAULT_SIGMA = {"QB": 0.25, "RB": 0.30, "WR": 0.35, "TE": 0.35, "DST": 0.30}
DEFAULT_CORR = 0.35            # share of the shock that is a common chalk lean


def resample_ownership(entities, rng, *, sigma=None, corr=DEFAULT_CORR):
    """Return a copy of ``entities`` with each ``own`` replaced by one draw.

    Each entity is drawn ``own * exp(shock)`` where ``shock`` mixes an
    idiosyncratic normal with a shared per-slate chalk shock:
        shock = sig * (sqrt(1-corr) * z_i + sqrt(corr) * z_slate)
    ``corr`` in [0, 1] controls how much the whole board moves together (0 =
    purely idiosyncratic). ``own`` is floored at a small positive value; the
    field build renormalizes per position afterward."""
    sigma = sigma or DEFAULT_SIGMA
    corr = float(min(max(corr, 0.0), 1.0))
    z_slate = float(rng.standard_normal())
    out = []
    for e in entities:
        sig = float(sigma.get(e.get("pos", ""), 0.30))
        z_i = float(rng.standard_normal())
        shock = sig * (np.sqrt(1.0 - corr) * z_i + np.sqrt(corr) * z_slate)
        d = dict(e)
        d["own"] = max(float(e.get("own", 0.0)) * float(np.exp(shock)), 1e-3)
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ents = [{"key": f"O{i}", "pos": p, "team": "AAA", "own": o}
            for i, (p, o) in enumerate(
                [("QB", 20), ("WR", 15), ("WR", 8), ("RB", 25),
                 ("TE", 5), ("DST", 10)])]
    # a draw preserves ordering-in-expectation but perturbs each value
    draws = [resample_ownership(ents, rng) for _ in range(4000)]
    means = {ents[i]["key"]: np.mean([d[i]["own"] for d in draws])
             for i in range(len(ents))}
    for i, e in enumerate(ents):
        m = means[e["key"]]
        # lognormal mean sits a touch above the point estimate; within ~15%
        assert abs(m - e["own"]) / e["own"] < 0.20, (e["own"], m)
        assert all(d[i]["own"] > 0 for d in draws)
    # correlated shock: with corr=1 all entities move the same direction vs base
    r = np.random.default_rng(1)
    d1 = resample_ownership(ents, r, corr=1.0)
    signs = [np.sign(d1[i]["own"] - ents[i]["own"]) for i in range(len(ents))]
    assert len(set(signs)) == 1, signs
    print("ownership_model.py self-test passed:",
          {k: round(v, 1) for k, v in means.items()})
