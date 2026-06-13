"""Deterministic selection tool (M5): blend per-criterion scores by weights, top-k.

score(x) = sum_i w_i * z_i(x), with z_i in [0,1] (normalized in state.compute_criteria).
This is the ONLY place acquisition decisions are made 鈥?fully auditable, no LLM in
the data path.
"""
from __future__ import annotations
import numpy as np
from .state import CRITERIA


def normalize_weights(weights, criteria=CRITERIA):
    w = {c: float(max(0.0, weights.get(c, 0.0))) for c in criteria}
    s = sum(w.values())
    if s <= 1e-12:
        return {c: 1.0 / len(criteria) for c in criteria}   # degenerate -> uniform
    return {c: v / s for c, v in w.items()}


def blended_scores(crit_scores, weights):
    w = normalize_weights(weights, list(crit_scores.keys()))
    n = len(next(iter(crit_scores.values())))
    total = np.zeros(n, dtype=np.float64)
    for c, z in crit_scores.items():
        total += w[c] * np.asarray(z, dtype=np.float64)
    return total


def select_topk(crit_scores, weights, budget, rng):
    scores = blended_scores(crit_scores, weights)
    budget = int(min(budget, len(scores)))
    # tiny random jitter to break ties deterministically-per-rng
    jitter = rng.random(len(scores)) * 1e-9
    order = np.argsort(-(scores + jitter))
    return order[:budget], scores
