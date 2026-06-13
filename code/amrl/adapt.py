"""Adaptation-control episode (PIVOT direction C1+C2).

Online AMR under drift with full supervision per step; the CONTROL TARGET is the
ADAPTATION ACTION each step (no_op / full / head-only finetune / reset+retrain /
recalibrate), with replay. A scheduler maps the IQ-free state (drift, ECE,
batch-accuracy, step) -> action. The non-LLM adaptation ORACLE = best schedule per
regime; the headroom gate asks whether it beats the best fixed schedule by >=3% AULC.
The LLM/agent (built only if the gate passes) would replace the scheduler.
"""
from __future__ import annotations
import math
import numpy as np

from .model import Classifier, accuracy, expected_calibration_error
from .regimes import Stream
from .metrics import summarize


def _grad_steps(n, batch_size, epochs):
    return epochs * max(1, math.ceil(n / batch_size))


def run_adapt_episode(cfg, data, scheduler, seed, verbose=False, replay_cap=4000):
    # Separate RNGs: the STREAM sequence must be identical across schedulers (paired),
    # so scheduler-dependent replay sampling must NOT advance the stream RNG.
    stream_rng = np.random.default_rng(seed)
    replay_rng = np.random.default_rng(seed + 999983)
    clf = Classifier(cfg, data.n_classes)
    stream = Stream(data, cfg, stream_rng)
    bs = cfg.model.batch_size
    base_ep = cfg.model.update_epochs

    Xs, ys = stream.seed_set()
    clf.fit(Xs, ys, cfg.model.warmup_epochs)
    replay_X, replay_y = [Xs], [ys]

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

    def _replay_sample(k):
        if k <= 0 or not replay_X:
            return None
        allX = np.concatenate(replay_X, axis=0)
        ally = np.concatenate(replay_y, axis=0)
        if k >= len(allX):
            return allX, ally
        idx = replay_rng.choice(len(allX), size=k, replace=False)
        return allX[idx], ally[idx]

    grad_steps = 0
    n_adapts = 0
    history = [{"step": -1, "labels_used": len(ys), **_eval(), "ece": float("nan"),
                "drift": 0.0, "action": "warmup"}]
    labels_used = len(ys)

    for t in range(stream.n_steps):
        step = stream.step_pool(t)
        Xb, yb = step.X, step.y
        # pre-update state signals (batch is new -> held-out-ish)
        pb = clf.predict_proba(Xb)
        batch_acc = accuracy(pb, yb)
        ece = expected_calibration_error(pb, yb, cfg.eval.ece_bins)
        state = {"drift": step.drift_level, "ece": ece, "batch_acc": batch_acc,
                 "step": t, "n_steps": stream.n_steps}
        stream.mark_used(step.gidx)              # consumed -> future batches stay fresh (held-out signals)
        act = scheduler(state)
        mode = act.get("mode", "no_op")
        ep = int(act.get("epochs", base_ep))
        rr = float(act.get("replay_ratio", 0.5))

        if mode in ("full", "head"):
            rs = _replay_sample(int(rr * len(Xb)))
            if rs is not None:
                trX = np.concatenate([Xb, rs[0]], axis=0)
                trY = np.concatenate([yb, rs[1]], axis=0)
            else:
                trX, trY = Xb, yb
            clf.fit(trX, trY, ep, head_only=(mode == "head"), lr=act.get("lr"))
            grad_steps += _grad_steps(len(trX), bs, ep)
            n_adapts += 1
        elif mode == "reset":
            clf.reset()
            rs = _replay_sample(replay_cap)
            trX = np.concatenate([Xb] + ([rs[0]] if rs is not None else []), axis=0)
            trY = np.concatenate([yb] + ([rs[1]] if rs is not None else []), axis=0)
            rep = max(ep, cfg.model.warmup_epochs)
            clf.fit(trX, trY, rep)
            grad_steps += _grad_steps(len(trX), bs, rep)
            n_adapts += 1
        elif mode == "recal":
            clf.recalibrate(Xb, yb)
            n_adapts += 1
        # else no_op

        # add batch to replay (capped)
        replay_X.append(Xb); replay_y.append(yb)
        tot = sum(len(a) for a in replay_X)
        while tot > replay_cap and len(replay_X) > 1:
            tot -= len(replay_X[1]); del replay_X[1]; del replay_y[1]
        labels_used += len(yb)

        e = _eval()
        history.append({"step": t, "labels_used": labels_used, "ece": ece,
                        "drift": step.drift_level, "action": mode, **e})
        if verbose:
            print(f"[step {t+1}/{stream.n_steps}] drift={step.drift_level:.2f} "
                  f"batch_acc={batch_acc:.2f} ece={ece:.3f} act={mode}({ep}) acc={e['test_acc']:.3f}")

    s = summarize(history, cfg.eval.target_acc)
    s["total_grad_steps"] = grad_steps
    s["n_adapts"] = n_adapts
    s["scheduler"] = getattr(scheduler, "name", "?")
    return {"history": history, "summary": s}


# ---- scheduler grid (non-LLM policies for the oracle) ------------------
def _named(fn, name):
    fn.name = name
    return fn


def scheduler_grid(cfg):
    ep = cfg.model.update_epochs
    G = {}
    G["never"] = _named(lambda s: {"mode": "no_op"}, "never")
    G["full1"] = _named(lambda s: {"mode": "full", "epochs": ep, "replay_ratio": 0.5}, "full1")
    G["head1"] = _named(lambda s: {"mode": "head", "epochs": ep, "replay_ratio": 0.5}, "head1")
    G["full2"] = _named(lambda s: ({"mode": "full", "epochs": ep, "replay_ratio": 0.5}
                                   if s["step"] % 2 == 0 else {"mode": "no_op"}), "full2")
    G["full3"] = _named(lambda s: ({"mode": "full", "epochs": ep, "replay_ratio": 0.5}
                                   if s["step"] % 3 == 0 else {"mode": "no_op"}), "full3")
    G["full_replay1"] = _named(lambda s: {"mode": "full", "epochs": ep, "replay_ratio": 1.0}, "full_replay1")
    G["full_replay0"] = _named(lambda s: {"mode": "full", "epochs": ep, "replay_ratio": 0.0}, "full_replay0")
    # state-conditioned ("controller-like") policies
    G["full_on_drift"] = _named(lambda s: ({"mode": "full", "epochs": ep, "replay_ratio": 0.5}
                                           if s["drift"] >= 0.5 else {"mode": "no_op"}), "full_on_drift")
    G["head_lo_full_hi"] = _named(lambda s: ({"mode": "head", "epochs": ep, "replay_ratio": 0.5}
                                             if s["drift"] < 0.5 else
                                             {"mode": "full", "epochs": ep, "replay_ratio": 1.0}), "head_lo_full_hi")
    G["reset_on_drift"] = _named(lambda s: ({"mode": "reset", "epochs": ep}
                                            if s["drift"] >= 0.5 else
                                            {"mode": "full", "epochs": ep, "replay_ratio": 0.5}), "reset_on_drift")
    G["adapt_on_accdrop"] = _named(lambda s: ({"mode": "full", "epochs": ep, "replay_ratio": 0.5}
                                              if s["batch_acc"] < 0.5 else {"mode": "no_op"}), "adapt_on_accdrop")
    G["recal_lo_full_hi"] = _named(lambda s: ({"mode": "recal"} if s["ece"] > 0.15 and s["drift"] < 0.5
                                              else {"mode": "full", "epochs": ep, "replay_ratio": 0.5}), "recal_lo_full_hi")
    # extra fixed baselines (so best-fixed is not understated)
    G["head2"] = _named(lambda s: ({"mode": "head", "epochs": ep, "replay_ratio": 0.5}
                                   if s["step"] % 2 == 0 else {"mode": "no_op"}), "head2")
    G["recal_only"] = _named(lambda s: {"mode": "recal"}, "recal_only")
    G["full_replay_big"] = _named(lambda s: {"mode": "full", "epochs": ep, "replay_ratio": 2.0}, "full_replay_big")
    # extra state-conditioned controllers
    G["full_on_ece"] = _named(lambda s: ({"mode": "full", "epochs": ep, "replay_ratio": 0.5}
                                         if (np.isfinite(s["ece"]) and s["ece"] > 0.15) else {"mode": "no_op"}), "full_on_ece")
    G["reset_on_accdrop"] = _named(lambda s: ({"mode": "reset", "epochs": ep} if s["batch_acc"] < 0.4
                                              else {"mode": "full", "epochs": ep, "replay_ratio": 0.5}), "reset_on_accdrop")
    return G


# which schedules are "fixed" (state-independent) vs state-conditioned controllers
FIXED = {"never", "full1", "head1", "full2", "full3", "full_replay1", "full_replay0",
         "head2", "recal_only", "full_replay_big"}
