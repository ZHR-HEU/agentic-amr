"""The online active-learning episode loop 鈥?ties M1-M5 together.

Per step: present unlabeled pool -> classifier produces proba/features ->
build IQ-free state cards -> controller emits criterion weights -> deterministic
selection queries top-k labels under budget -> 20% held out as an ECE calibration
reservoir (never trained on) -> incrementally update classifier -> evaluate.
"""
from __future__ import annotations
import numpy as np

from .model import Classifier, accuracy, expected_calibration_error
from .regimes import Stream
from .state import compute_criteria, build_cards
from .selection import select_topk, normalize_weights
from .metrics import summarize


def _counts(y, n_classes):
    c = np.bincount(np.asarray(y, dtype=int), minlength=n_classes)
    return c.astype(np.int64)


def run_episode(cfg, data, controller, rng, verbose=True):
    n_classes = data.n_classes
    clf = Classifier(cfg, n_classes)
    stream = Stream(data, cfg, rng)
    budget = stream.budget_per_step
    calib_frac = 0.2

    # --- warmup on seed labeled set ---
    Xs, ys = stream.seed_set()
    labeled_X, labeled_y = Xs.copy(), ys.copy()
    train_counts = _counts(labeled_y, n_classes)     # counts of TRAINED labels only
    clf.fit(labeled_X, labeled_y, cfg.model.warmup_epochs)

    calib_X = np.zeros((0,) + Xs.shape[1:], dtype=np.float32)
    calib_y = np.zeros((0,), dtype=np.int64)

    Xte, yte = data.X[data.test_idx], data.y[data.test_idx]
    te_snr = data.snr[data.test_idx]
    bands = {"low": te_snr <= -6, "mid": (te_snr > -6) & (te_snr < 8), "high": te_snr >= 8}

    def _eval():
        p = clf.predict_proba(Xte)
        pred = p.argmax(axis=1)
        d = {"test_acc": float((pred == yte).mean())}
        for b, m in bands.items():
            d[f"acc_{b}"] = float((pred[m] == yte[m]).mean()) if m.sum() > 0 else float("nan")
        return d

    labels_used = len(labeled_y)
    e0 = _eval()
    history = [{"step": -1, "labels_used": labels_used, "ece": float("nan"),
                "drift": 0.0, "target_snr": float("nan"), "weights": {}, **e0}]
    if verbose:
        print(f"[warmup] labels={labels_used} test_acc={e0['test_acc']:.3f}")

    for t in range(stream.n_steps):
        step = stream.step_pool(t)
        proba = clf.predict_proba(step.X)
        feats = clf.features(step.X)
        labeled_feats = clf.features(labeled_X)
        crit = compute_criteria(proba, feats, labeled_feats, train_counts, n_classes, rng)

        if len(calib_y) >= 20:
            ece = expected_calibration_error(clf.predict_proba(calib_X), calib_y,
                                             cfg.eval.ece_bins)
        else:
            ece = float("nan")

        rf, cand = build_cards(t, stream.n_steps, step, proba, crit, train_counts,
                               ece, budget, n_classes)
        w = controller.weights(rf, cand)
        sel, scores = select_topk(crit, w, budget, rng)

        sel_X, sel_y = step.X[sel], step.y[sel]
        stream.mark_used(step.gidx[sel])             # queried -> never re-presented / re-trained
        n_calib = int(round(calib_frac * len(sel)))
        perm = rng.permutation(len(sel))
        ci, ti = perm[:n_calib], perm[n_calib:]
        calib_X = np.concatenate([calib_X, sel_X[ci]], axis=0)   # held out from training
        calib_y = np.concatenate([calib_y, sel_y[ci]], axis=0)
        labeled_X = np.concatenate([labeled_X, sel_X[ti]], axis=0)
        labeled_y = np.concatenate([labeled_y, sel_y[ti]], axis=0)
        train_counts = train_counts + _counts(sel_y[ti], n_classes)  # only TRAINED labels
        labels_used += len(sel)                       # budget spent = all queried

        clf.fit(labeled_X, labeled_y, cfg.model.update_epochs)
        e = _eval()
        nz = normalize_weights(w, list(crit.keys()))
        history.append({"step": t, "labels_used": labels_used, "ece": ece,
                        "drift": step.drift_level, "target_snr": step.target_snr,
                        "weights": nz, **e})
        if verbose:
            top = max(nz, key=nz.get)
            ece_s = f"{ece:.3f}" if np.isfinite(ece) else "nan"
            print(f"[step {t+1}/{stream.n_steps}] snr~{step.target_snr:.0f} drift={step.drift_level:.2f} "
                  f"ece={ece_s} labels={labels_used} acc={e['test_acc']:.3f} top_w={top}")

    summary = summarize(history, cfg.eval.target_acc)
    summary["controller"] = controller.name
    if hasattr(controller, "n_fallback"):
        summary["llm_calls"] = getattr(controller, "n_calls", 0)
        summary["llm_fallbacks"] = controller.n_fallback
    return {"history": history, "summary": summary}
