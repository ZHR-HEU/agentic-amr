#!/usr/bin/env python
"""B1b: ADAPTIVE-oracle diagnostic 鈥?does a STATE-CONDITIONED switching policy have
headroom over the best STATIC mixture? (B1 only tested static mixtures, but the
flagship's thesis is that weights should ADAPT to state / calibration.)

Per regime, pick an adaptive axis (snr | drift | phase), bin state into 2 bins, and
search the 25 policies = menu M (bin0) x menu M (bin1). The 5 homogeneous policies
ARE static mixtures; the 20 heterogeneous ones are adaptive. PAIRED episodes.

  adaptive_oracle = best of 25 ; static_oracle = best of 5 homogeneous.
  ADAPTIVITY HAS HEADROOM iff mean(adaptive - static) >= floor AND (>=3 seeds) CI>0.

Outputs the winning adaptive policy per regime (e.g. low-SNR->coreset, high-SNR->entropy
would directly validate calibration-gating).
"""
import argparse
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from amrl.config import load_config
from amrl.data import load_dataset
from amrl.controllers import BinPolicyController
from amrl.episode import run_episode

ALL_REGIMES = ["snr_ramp", "snr_step", "channel_drift", "class_emergence", "mixed"]
AXIS = {"snr_ramp": "snr", "snr_step": "snr", "channel_drift": "drift",
        "mixed": "drift", "class_emergence": "phase"}
MENU = [
    ("entropy", {"entropy": 1.0}),
    ("coreset", {"coreset": 1.0}),
    ("class_balance", {"class_balance": 1.0}),
    ("random", {"random": 1.0}),
    ("fixed_hybrid", {"entropy": .4, "margin": .2, "coreset": .2, "class_balance": .2}),
]
GAP_FLOOR = 0.005


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def mean_ci(vals):
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    m = float(a.mean())
    if n < 2:
        return m, float("nan")
    se = float(a.std(ddof=1)) / np.sqrt(n)
    return m, 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2016")
    ap.add_argument("--regimes", default=",".join(ALL_REGIMES))
    ap.add_argument("--seeds", default="42,202,303")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    policies = list(itertools.product(MENU, MENU))   # 25 (bin0, bin1)

    base_ov = [f"dataset.name={args.dataset}",
               "stream.n_steps=15", "stream.pool_size=600",
               "stream.budget_per_step=30", "stream.seed_labeled=300",
               "model.warmup_epochs=20", "model.update_epochs=5"]
    if args.smoke:
        base_ov = [f"dataset.name={args.dataset}", "stream.n_steps=2",
                   "stream.pool_size=200", "stream.budget_per_step=15",
                   "stream.seed_labeled=120", "model.warmup_epochs=2", "model.update_epochs=1"]
        seeds = seeds[:1]
        policies = policies[:4]
    cfg = load_config(args.config, base_ov)

    print(f"== B1b adaptive == dataset={args.dataset} regimes={regimes} seeds={seeds} "
          f"policies={len(policies)}", flush=True)

    # store[regime][policy_label] = [aulc over seeds]
    store = {r: {f"{a[0]}|{b[0]}": [] for a, b in policies} for r in regimes}
    n_ep, total = 0, len(seeds) * len(regimes) * len(policies)

    for seed in seeds:
        cfg.seed = seed
        set_seed(seed)
        data = load_dataset(cfg)
        print(f"[seed {seed}] data X={data.X.shape}", flush=True)
        for regime in regimes:
            cfg.stream.regime = regime
            axis = AXIS[regime]
            for (a, b) in policies:
                lab = f"{a[0]}|{b[0]}"
                set_seed(seed)
                rng = np.random.default_rng(seed)
                ctrl = BinPolicyController(cfg, rng, {0: a[1], 1: b[1]}, axis, lab)
                s = run_episode(cfg, data, ctrl, rng, verbose=False)["summary"]
                store[regime][lab].append(s["aulc"])
                n_ep += 1
            print(f"  [{regime}/{axis}] ({n_ep}/{total}) done", flush=True)

    report = {"dataset": args.dataset, "seeds": seeds, "gap_floor": GAP_FLOOR, "regimes": {}}
    helped = 0
    for regime in regimes:
        means = {L: float(np.mean(store[regime][L])) for L in store[regime]}
        homog = {L: m for L, m in means.items() if L.split("|")[0] == L.split("|")[1]}
        static_best = max(homog, key=homog.get)
        adapt_best = max(means, key=means.get)
        is_adaptive = adapt_best.split("|")[0] != adapt_best.split("|")[1]
        gaps = [store[regime][adapt_best][i] - store[regime][static_best][i]
                for i in range(len(seeds))]
        mgap, ci = mean_ci(gaps)
        lower = mgap - ci if np.isfinite(ci) else float("nan")
        helps = bool(is_adaptive and mgap >= GAP_FLOOR
                     and (len(seeds) < 3 or (np.isfinite(lower) and lower > 0)))
        helped += int(helps)
        report["regimes"][regime] = {
            "axis": AXIS[regime], "adaptive_best": adapt_best, "adaptive_aulc": means[adapt_best],
            "static_best": static_best, "static_aulc": means[static_best],
            "headroom": mgap, "headroom_ci95_lower": lower, "is_adaptive": is_adaptive,
            "adaptivity_helps": helps, "means": means,
        }
        print(f"[ADAPT {regime}/{AXIS[regime]}] adaptive={adapt_best}({means[adapt_best]:.4f}) "
              f"static={static_best}({means[static_best]:.4f}) headroom={mgap:+.4f} "
              f"ci_low={lower:+.4f} -> {'HELPS' if helps else 'no'}", flush=True)

    report["regimes_adaptivity_helps"] = helped
    report["regimes_total"] = len(regimes)
    out_dir = os.path.join(ROOT, cfg.eval.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = "smoke_" if args.smoke else ""
    with open(os.path.join(out_dir, f"b1b_adaptive_{tag}{args.dataset}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[B1b SUMMARY {args.dataset}] adaptivity_helps={helped}/{len(regimes)}", flush=True)
    open(os.path.join(out_dir, f"B1B_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
