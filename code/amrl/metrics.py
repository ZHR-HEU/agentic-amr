"""Episode metrics: label-efficiency AULC (overall + per-SNR-band), labels-to-target.

SNR-banded AULC (low/mid/high) is reported so the B1 gate is not hidden by
averaging over unlearnable low-SNR samples (cross-model methodology review).
(Post-drift recovery lag and regret-to-oracle are added in B2/B4.)
"""
from __future__ import annotations
import numpy as np


def summarize(history, target_acc):
    labels = np.array([h["labels_used"] for h in history], dtype=np.float64)

    def _aulc(key):
        vals = np.array([h.get(key, np.nan) for h in history], dtype=np.float64)
        mask = ~np.isnan(vals)
        if mask.sum() == 0:
            return float("nan")
        L, V = labels[mask], vals[mask]
        span = L[-1] - L[0]
        return float(np.trapz(V, L) / span) if (span > 0 and len(L) > 1) else float(V.mean())

    acc = np.array([h["test_acc"] for h in history], dtype=np.float64)
    to_target = None
    for h in history:
        if h["test_acc"] >= target_acc:
            to_target = int(h["labels_used"])
            break
    return {
        "aulc": _aulc("test_acc"),
        "aulc_low": _aulc("acc_low"),
        "aulc_mid": _aulc("acc_mid"),
        "aulc_high": _aulc("acc_high"),
        "final_acc": float(acc[-1]),
        "final_acc_mid": float(history[-1].get("acc_mid", float("nan"))),
        "final_acc_high": float(history[-1].get("acc_high", float("nan"))),
        "labels_to_target": to_target,
        "target_acc": target_acc,
        "total_labels": int(labels[-1]),
        "n_points": len(history),
    }
