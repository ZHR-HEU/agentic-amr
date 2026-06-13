"""State cards (M3) + acquisition criteria.

All quantities here are deterministic summaries derived from the classifier's
outputs/features and label bookkeeping 鈥?NO raw IQ is exposed to the controller.
Each criterion returns a per-pool score in [0,1] where HIGHER = more desirable to
acquire; the selection engine blends them by the controller's weights.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

CRITERIA = ["entropy", "margin", "coreset", "class_balance", "random"]


def _minmax(v):
    v = np.asarray(v, dtype=np.float64)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def compute_criteria(proba, feats, labeled_feats, labeled_counts, n_classes, rng):
    """Return {criterion: np.array(pool,) normalized to [0,1], higher=better}."""
    n = len(proba)
    eps = 1e-12
    # entropy (normalized by log C)
    ent = -(proba * np.log(proba + eps)).sum(axis=1) / np.log(n_classes)
    # margin uncertainty: 1 - (p1 - p2)
    part = np.sort(proba, axis=1)
    margin_unc = 1.0 - (part[:, -1] - part[:, -2])
    # coreset / diversity: min feature distance to labeled set (higher=more novel).
    # Memory-bounded: cap reference set, use |a-b|^2 = |a|^2+|b|^2-2a.b (no 3-D broadcast).
    if labeled_feats is not None and len(labeled_feats) > 0:
        lf = np.asarray(labeled_feats, dtype=np.float64)
        MAX_REF = 1500
        if len(lf) > MAX_REF:
            lf = lf[rng.choice(len(lf), size=MAX_REF, replace=False)]
        lf_sq = (lf ** 2).sum(axis=1)
        F = np.asarray(feats, dtype=np.float64)
        d_min = np.empty(n)
        for i in range(0, n, 256):
            chunk = F[i:i + 256]
            sq = (chunk ** 2).sum(axis=1)[:, None] + lf_sq[None, :] - 2.0 * (chunk @ lf.T)
            np.maximum(sq, 0.0, out=sq)
            d_min[i:i + 256] = np.sqrt(sq.min(axis=1))
        coreset = d_min
    else:
        coreset = np.ones(n)
    # class balance: prefer predicted-class that is under-represented in buffer
    pred = proba.argmax(axis=1)
    counts = np.asarray(labeled_counts, dtype=np.float64)
    inv = 1.0 / (counts + 1.0)
    class_bal = inv[pred]
    # random
    rnd = rng.random(n)
    return {
        "entropy": _minmax(ent),
        "margin": _minmax(margin_unc),
        "coreset": _minmax(coreset),
        "class_balance": _minmax(class_bal),
        "random": rnd,
    }


@dataclass
class RFStateCard:
    step: int
    n_steps: int
    snr_est: float
    snr_std: float
    drift_level: float
    ece: float                      # may be nan (insufficient calibration data)
    budget_left: int
    budget_total: int
    class_counts: list              # labeled buffer counts per class
    n_classes: int

    def ece_text(self):
        if not np.isfinite(self.ece):
            return "unknown(insufficient calib data)"
        tag = "high/miscalibrated" if self.ece > 0.15 else ("moderate" if self.ece > 0.07 else "low/well-calibrated")
        return f"{self.ece:.3f}({tag})"

    def to_text(self):
        zero = sum(1 for c in self.class_counts if c == 0)
        return (
            f"RF State Card:\n"
            f"- step: {self.step+1}/{self.n_steps}\n"
            f"- SNR_est: {self.snr_est:.1f} dB (std {self.snr_std:.1f})\n"
            f"- drift_level: {self.drift_level:.2f} (0=stable,1=max channel drift)\n"
            f"- classifier_ECE: {self.ece_text()}\n"
            f"- labeled_class_counts: {self.class_counts} ({zero}/{self.n_classes} classes still unseen)\n"
            f"- label_budget_left_this_step: {self.budget_left}/{self.budget_total}\n"
        )

    def to_vector(self):
        counts = np.asarray(self.class_counts, dtype=np.float64)
        tot = counts.sum() + 1.0
        ece = self.ece if np.isfinite(self.ece) else 0.0
        return np.array([
            self.step / max(1, self.n_steps - 1),
            self.snr_est / 30.0,
            self.snr_std / 20.0,
            self.drift_level,
            ece,
            (counts == 0).mean(),                  # fraction of unseen classes
            counts.std() / tot,                    # imbalance
        ], dtype=np.float64)


@dataclass
class CandidateCard:
    pool_size: int
    entropy_mean: float
    entropy_std: float
    margin_unc_mean: float
    coverage_gap: float             # mean normalized coreset score
    class_deficit: list             # under-represented classes (by predicted label)
    n_classes: int

    def to_text(self):
        return (
            f"Candidate Card (unlabeled pool):\n"
            f"- pool_size: {self.pool_size}\n"
            f"- entropy_mean: {self.entropy_mean:.2f} (std {self.entropy_std:.2f})\n"
            f"- margin_uncertainty_mean: {self.margin_unc_mean:.2f}\n"
            f"- feature_coverage_gap: {self.coverage_gap:.2f} (higher=pool covers regions far from labeled)\n"
            f"- predicted under-represented classes: {self.class_deficit}\n"
        )

    def to_vector(self):
        return np.array([
            self.entropy_mean, self.entropy_std,
            self.margin_unc_mean, self.coverage_gap,
            len(self.class_deficit) / max(1, self.n_classes),
        ], dtype=np.float64)


def build_cards(step, n_steps, stepdata, proba, crit, labeled_counts,
                ece, budget_total, n_classes):
    snr = stepdata.snr
    counts = list(int(c) for c in labeled_counts)
    deficit = [i for i, c in enumerate(counts) if c <= np.median(counts)]
    rf = RFStateCard(
        step=step, n_steps=n_steps,
        snr_est=float(np.mean(snr)), snr_std=float(np.std(snr)),
        drift_level=stepdata.drift_level, ece=ece,
        budget_left=budget_total, budget_total=budget_total,
        class_counts=counts, n_classes=n_classes,
    )
    # entropy_mean/std reported on the normalized criterion for stability
    cand = CandidateCard(
        pool_size=len(proba),
        entropy_mean=float(np.mean(crit["entropy"])),
        entropy_std=float(np.std(crit["entropy"])),
        margin_unc_mean=float(np.mean(crit["margin"])),
        coverage_gap=float(np.mean(crit["coreset"])),
        class_deficit=deficit[:8],
        n_classes=n_classes,
    )
    return rf, cand
