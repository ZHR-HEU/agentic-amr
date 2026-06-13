"""Streaming / drift simulator (M2): deterministic non-stationary episodes.

Regimes: snr_ramp, snr_step, channel_drift, class_emergence, mixed.
Each step yields a pool of `pool_size` unlabeled candidates (with hidden true
labels + SNR + global indices). Queried samples are marked "used" and excluded
from future pools, which guarantees the ECE calibration reservoir is never
trained on and trained samples are never re-presented (Codex review fix).
Channel drift is synthesized on IQ (gain, phase, CFO, extra noise) with
parameters that evolve over steps 鈥?RML data is read in place, never modified.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StepData:
    X: np.ndarray            # (pool_size, 2, L)  possibly channel-perturbed
    y: np.ndarray            # (pool_size,) hidden true labels
    snr: np.ndarray          # (pool_size,)
    gidx: np.ndarray         # (pool_size,) global indices into data.X
    step: int
    target_snr: float
    drift_level: float
    active_classes: list


def apply_channel(X, gain=1.0, phase=0.0, cfo=0.0, noise_std=0.0, rng=None):
    """Apply a synthetic channel to IQ signals. X: (n,2,L) -> (n,2,L)."""
    n, _, L = X.shape
    iq = X[:, 0, :] + 1j * X[:, 1, :]
    nidx = np.arange(L)
    rot = np.exp(1j * (phase + 2 * np.pi * cfo * nidx))[None, :]
    iq = gain * iq * rot
    out = np.empty_like(X)
    out[:, 0, :] = iq.real
    out[:, 1, :] = iq.imag
    if noise_std > 0 and rng is not None:
        out = out + rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    return out.astype(np.float32)


class Stream:
    def __init__(self, data, cfg, rng):
        self.data = data
        self.cfg = cfg
        self.rng = rng
        s = cfg.stream
        self.regime = s.regime
        self.n_steps = s.n_steps
        self.pool_size = s.pool_size
        self.budget_per_step = s.budget_per_step
        self.warmup_snr = tuple(s.warmup_snr)
        self.seed_labeled = s.seed_labeled
        self.n_classes = data.n_classes

        self.tr = data.train_idx
        self.tr_snr = data.snr[self.tr]
        self.tr_y = data.y[self.tr]
        self.snr_values = np.array(sorted(np.unique(self.tr_snr)))
        # position lookup + used-mask for train/calib isolation
        self.g2pos = {int(g): i for i, g in enumerate(self.tr)}
        self.used_mask = np.zeros(len(self.tr), dtype=bool)

    # ---- isolation bookkeeping ----------------------------------------
    def mark_used(self, gidx_array):
        for g in np.asarray(gidx_array).reshape(-1):
            p = self.g2pos.get(int(g))
            if p is not None:
                self.used_mask[p] = True

    def _sample_positions(self, mask, m):
        avail = mask & (~self.used_mask)
        idx_local = np.where(avail)[0]
        if len(idx_local) == 0:                          # relax constraints, keep isolation
            idx_local = np.where(~self.used_mask)[0]
        if len(idx_local) == 0:                          # extreme fallback (tiny pools only)
            idx_local = np.arange(len(self.tr))
        replace = m > len(idx_local)
        return self.rng.choice(idx_local, size=m, replace=replace)

    def _gather_positions(self, pos):
        gidx = self.tr[pos]
        return (self.data.X[gidx].copy(), self.data.y[gidx].copy(),
                self.data.snr[gidx].copy(), gidx)

    # ---- seed labeled set (warmup) ------------------------------------
    def seed_set(self):
        lo, hi = self.warmup_snr
        mask = (self.tr_snr >= lo) & (self.tr_snr <= hi)
        pos = self._sample_positions(mask, self.seed_labeled)
        X, y, _, gidx = self._gather_positions(pos)
        self.mark_used(gidx)             # seed is trained -> never re-present / re-use
        return X, y

    # ---- per-step regime schedules ------------------------------------
    def _target_snr(self, t):
        lo, hi = float(self.snr_values.min()), float(self.snr_values.max())
        frac = t / max(1, self.n_steps - 1)
        tri = 1.0 - abs(2 * frac - 1.0)              # 0 -> 1 -> 0
        if self.regime == "snr_ramp":
            return lo + (hi - lo) * tri
        if self.regime == "snr_step":
            return lo + 4 if frac < 0.5 else hi - 4
        return 6.0

    def _active_classes(self, t):
        if self.regime in ("class_emergence", "mixed"):
            half = (self.n_classes + 1) // 2
            if t < self.n_steps // 2:
                return list(range(half))
            return list(range(self.n_classes))
        return list(range(self.n_classes))

    def _drift_level(self, t):
        frac = t / max(1, self.n_steps - 1)
        if self.regime in ("channel_drift", "mixed"):
            return float(frac)
        return 0.0

    def step_pool(self, t) -> StepData:
        target = self._target_snr(t)
        active = self._active_classes(t)
        drift = self._drift_level(t)

        snr_win = 4
        snr_mask = np.abs(self.tr_snr - target) <= snr_win
        class_mask = np.isin(self.tr_y, active)
        mask = snr_mask & class_mask
        if (mask & ~self.used_mask).sum() < max(1, self.pool_size // 4):
            mask = class_mask                        # widen if too sparse
        pos = self._sample_positions(mask, self.pool_size)
        X, y, snr, gidx = self._gather_positions(pos)

        if drift > 0:
            X = apply_channel(
                X,
                gain=1.0 - 0.3 * drift,
                phase=np.pi * drift,
                cfo=0.02 * drift / X.shape[2],
                noise_std=0.05 * drift,
                rng=self.rng,
            )
        return StepData(X, y, snr, gidx, t, float(target), float(drift), active)
