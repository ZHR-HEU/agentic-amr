#!/usr/bin/env python
"""Adaptation-oracle headroom gate for the pivoted direction (C1+C2).

Does choosing the ADAPTATION SCHEDULE (per regime / state) have real headroom over
the best FIXED schedule? Sweep the scheduler grid on PAIRED episodes; per regime,
oracle = best schedule, baseline = best FIXED schedule (global). Commit to building
the LLM adaptation controller ONLY if:
    oracle - baseline >= GATE_AULC (default 0.03) in >= MIN_REGIMES regimes,
    with paired CI>0 (>=3 seeds), on BOTH datasets.
(High bar by design 鈥?cross-model review: adaptation should yield multi-point swings,
unlike acquisition-criterion weighting which had sub-1% oracle headroom.)

Usage: python scripts/run_b2a_adaptoracle.py --dataset rml2016 --seeds 42,202,303
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from scipy import stats

from amrl.config import load_config
from amrl.data import load_dataset
from amrl.adapt import run_adapt_episode, scheduler_grid, FIXED

ALL_REGIMES = ["snr_ramp", "snr_step", "channel_drift", "class_emergence", "mixed"]
GATE_AULC = 0.03
MIN_REGIMES = 3


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
    return m, float(stats.t.ppf(0.975, n - 1)) * se   # t-based 95% CI


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

    base_ov = [f"dataset.name={args.dataset}",
               "stream.n_steps=15", "stream.pool_size=400",
               "stream.budget_per_step=30", "stream.seed_labeled=300",
               "model.warmup_epochs=20", "model.update_epochs=5"]
    if args.smoke:
        base_ov = [f"dataset.name={args.dataset}", "stream.n_steps=3",
                   "stream.pool_size=200", "stream.seed_labeled=150",
                   "model.warmup_epochs=3", "model.update_epochs=2"]
        seeds = seeds[:1]
    cfg = load_config(args.config, base_ov)
    grid = scheduler_grid(cfg)
    if args.smoke:
        grid = {k: grid[k] for k in ["never", "full1", "head_lo_full_hi", "reset_on_drift"]}
    labels = list(grid.keys())

    print(f"== B2a adapt-oracle == dataset={args.dataset} regimes={regimes} seeds={seeds} "
          f"schedulers={len(labels)} GATE>={GATE_AULC}", flush=True)

    store = {r: {L: {"aulc": [], "grad": []} for L in labels} for r in regimes}
    n_ep, total = 0, len(seeds) * len(regimes) * len(labels)
    for seed in seeds:
        cfg.seed = seed
        set_seed(seed)
        data = load_dataset(cfg)
        print(f"[seed {seed}] data X={data.X.shape}", flush=True)
        for regime in regimes:
            cfg.stream.regime = regime
            for L in labels:
                set_seed(seed)                       # torch model-init parity across schedulers
                out = run_adapt_episode(cfg, data, grid[L], seed, verbose=False)["summary"]
                store[regime][L]["aulc"].append(out["aulc"])
                store[regime][L]["grad"].append(out["total_grad_steps"])
                n_ep += 1
            bestL = max(store[regime], key=lambda L: np.mean(store[regime][L]["aulc"]))
            print(f"  [{regime}] ({n_ep}/{total}) best={bestL} "
                  f"aulc={np.mean(store[regime][bestL]['aulc']):.4f}", flush=True)

    # baseline = best FIXED schedule by mean over all regimes & seeds
    fixed_mean = {}
    for L in labels:
        if L in FIXED:
            vals = [v for r in regimes for v in store[r][L]["aulc"]]
            fixed_mean[L] = float(np.mean(vals))
    baseline = max(fixed_mean, key=fixed_mean.get)

    report = {"dataset": args.dataset, "seeds": seeds, "gate_aulc": GATE_AULC,
              "baseline_fixed": baseline, "regimes": {}}
    ctrl_labels = [L for L in labels if L not in FIXED]
    passed = 0
    for regime in regimes:
        means = {L: float(np.mean(store[regime][L]["aulc"])) for L in labels}

        def _gap(orac):
            gaps = [store[regime][orac]["aulc"][i] - store[regime][baseline]["aulc"][i]
                    for i in range(len(seeds))]
            mg, ci = mean_ci(gaps)
            return mg, (mg - ci if np.isfinite(ci) else float("nan"))

        any_oracle = max(means, key=means.get)
        ag, ag_low = _gap(any_oracle)
        # PRIMARY gate: a state-conditioned CONTROLLER must beat best-fixed (not just per-regime tuning)
        ctrl_oracle = max(ctrl_labels, key=lambda L: means[L]) if ctrl_labels else None
        cg, cg_low = _gap(ctrl_oracle) if ctrl_oracle else (float("nan"), float("nan"))
        gate = bool(ctrl_oracle and cg >= GATE_AULC
                    and (len(seeds) < 3 or (np.isfinite(cg_low) and cg_low > 0)))
        passed += int(gate)
        report["regimes"][regime] = {
            "controller_oracle": ctrl_oracle,
            "controller_aulc": (means[ctrl_oracle] if ctrl_oracle else None),
            "controller_gain": cg, "controller_gain_ci_lower": cg_low, "gate_pass": gate,
            "any_oracle": any_oracle, "any_oracle_is_controller": any_oracle not in FIXED,
            "any_oracle_aulc": means[any_oracle], "any_gain": ag,
            "baseline_aulc": means[baseline],
            "ctrl_grad": (float(np.mean(store[regime][ctrl_oracle]["grad"])) if ctrl_oracle else None),
            "baseline_grad": float(np.mean(store[regime][baseline]["grad"])),
            "means": means,
        }
        print(f"[GATE {regime}] ctrl={ctrl_oracle}({means.get(ctrl_oracle, float('nan')):.4f}) "
              f"base={baseline}({means[baseline]:.4f}) ctrl_gain={cg:+.4f} ci_low={cg_low:+.4f} | "
              f"any={any_oracle}({means[any_oracle]:.4f}{'*ctrl' if any_oracle not in FIXED else ''}) "
              f"-> {'PASS' if gate else 'fail'}", flush=True)

    report["regimes_passed"] = passed
    report["regimes_total"] = len(regimes)
    report["dataset_gate"] = bool(passed >= MIN_REGIMES)
    out_dir = os.path.join(ROOT, cfg.eval.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = "smoke_" if args.smoke else ""
    with open(os.path.join(out_dir, f"b2a_adaptoracle_{tag}{args.dataset}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[B2a SUMMARY {args.dataset}] passed={passed}/{len(regimes)} "
          f"dataset_gate={'PASS' if report['dataset_gate'] else 'FAIL'} baseline={baseline}", flush=True)
    open(os.path.join(out_dir, f"B2A_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
