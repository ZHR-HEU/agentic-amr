#!/usr/bin/env python
"""B1: oracle soft-mixture envelope + C1 headroom decision gate.

Per regime, sweep a grid of STATIC criterion-mixtures on PAIRED episodes (within a
seed: same data split / stream / model init; only the mixture differs). The static
"best mixture in hindsight" is an OPTIMISTIC oracle, so the gate is PAIRED-STATISTICAL
(per-seed gap of the oracle mixture vs the best single / uniform, with a CI) rather
than a bare threshold (cross-model methodology review). We also report SNR-banded
AULC (low/mid/high) so the gate is not hidden by unlearnable low-SNR dilution.

C1 GATE (per regime): oracle is a true MIX (not corner/uniform) AND mean paired gap
>= floor AND (with >=3 seeds) the gap's 95% CI excludes 0.

Usage:
  python scripts/run_b1_oracle.py --dataset rml2016 --seeds 42,202,303
  python scripts/run_b1_oracle.py --dataset rml2016 --smoke
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from amrl.config import load_config
from amrl.data import load_dataset
from amrl.controllers import StaticMixture
from amrl.episode import run_episode
from amrl.state import CRITERIA

ALL_REGIMES = ["snr_ramp", "snr_step", "channel_drift", "class_emergence", "mixed"]
GAP_FLOOR = 0.005


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _compositions(total, bins):
    if bins == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _compositions(total - i, bins - 1):
            yield (i,) + rest


def build_grid():
    """corners(single) + pairwise + uniform + fixed_hybrid + curated 3/4-way blends."""
    grid = []
    for comp in _compositions(2, len(CRITERIA)):        # 15: corners + pairwise
        nz = [CRITERIA[i] for i, c in enumerate(comp) if c > 0]
        w = {CRITERIA[i]: comp[i] / 2.0 for i in range(len(CRITERIA))}
        if len(nz) == 1:
            grid.append({"label": f"single:{nz[0]}", "kind": "single", "w": w})
        else:
            grid.append({"label": f"pair:{nz[0]}+{nz[1]}", "kind": "mix", "w": w})
    grid.append({"label": "uniform", "kind": "uniform", "w": {c: 0.2 for c in CRITERIA}})
    grid.append({"label": "fixed_hybrid", "kind": "mix",
                 "w": {"entropy": 0.4, "margin": 0.2, "coreset": 0.2,
                       "class_balance": 0.2, "random": 0.0}})
    curated = [
        ("tri:ent+core+bal", {"entropy": .34, "coreset": .33, "class_balance": .33}),
        ("tri:ent+mar+core", {"entropy": .34, "margin": .33, "coreset": .33}),
        ("tri:mar+core+bal", {"margin": .34, "coreset": .33, "class_balance": .33}),
        ("quad:emcb", {"entropy": .25, "margin": .25, "coreset": .25, "class_balance": .25}),
        ("blend:ent50+core25+bal25", {"entropy": .5, "coreset": .25, "class_balance": .25}),
        ("blend:ent40+core30+bal30", {"entropy": .4, "coreset": .3, "class_balance": .3}),
    ]
    for lab, w in curated:
        full = {c: float(w.get(c, 0.0)) for c in CRITERIA}
        grid.append({"label": lab, "kind": "mix", "w": full})
    return grid


def mean_ci(vals):
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    m = float(a.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    se = float(a.std(ddof=1)) / np.sqrt(n)
    return m, se, 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2016")
    ap.add_argument("--regimes", default=",".join(ALL_REGIMES))
    ap.add_argument("--seeds", default="42,202,303")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    grid = build_grid()
    if args.only:
        keep = set(x.strip() for x in args.only.split(",") if x.strip())
        grid = [g for g in grid if g["label"] in keep]
    if args.limit:
        grid = grid[:args.limit]
    kind = {g["label"]: g["kind"] for g in grid}

    base_ov = [
        f"dataset.name={args.dataset}",
        "stream.n_steps=15", "stream.pool_size=600",
        "stream.budget_per_step=30", "stream.seed_labeled=300",
        "model.warmup_epochs=20", "model.update_epochs=5",
    ]
    if args.smoke:
        base_ov = [f"dataset.name={args.dataset}", "stream.n_steps=2",
                   "stream.pool_size=200", "stream.budget_per_step=15",
                   "stream.seed_labeled=120", "model.warmup_epochs=2",
                   "model.update_epochs=1"]
        seeds = seeds[:1]
    cfg = load_config(args.config, base_ov + list(args.overrides))

    print(f"== B1 oracle == dataset={args.dataset} regimes={regimes} seeds={seeds} "
          f"mixtures={len(grid)}", flush=True)

    # store[regime][label] = {"aulc":[...], "aulc_mid":[...], "aulc_high":[...]} aligned by seed
    store = {r: {g["label"]: {"aulc": [], "aulc_mid": [], "aulc_high": []} for g in grid}
             for r in regimes}
    n_ep, total = 0, len(seeds) * len(regimes) * len(grid)

    for seed in seeds:
        cfg.seed = seed
        set_seed(seed)
        data = load_dataset(cfg)
        print(f"[seed {seed}] data X={data.X.shape} test={len(data.test_idx)}", flush=True)
        for regime in regimes:
            cfg.stream.regime = regime
            for g in grid:
                set_seed(seed)                       # PAIRED across mixtures
                rng = np.random.default_rng(seed)
                ctrl = StaticMixture(cfg, rng, g["w"], g["label"])
                s = run_episode(cfg, data, ctrl, rng, verbose=False)["summary"]
                store[regime][g["label"]]["aulc"].append(s["aulc"])
                store[regime][g["label"]]["aulc_mid"].append(s["aulc_mid"])
                store[regime][g["label"]]["aulc_high"].append(s["aulc_high"])
                n_ep += 1
            bestL = max(store[regime], key=lambda L: np.mean(store[regime][L]["aulc"]))
            print(f"  [{regime}] ({n_ep}/{total}) best={bestL} "
                  f"aulc={np.mean(store[regime][bestL]['aulc']):.4f}", flush=True)

    # ---- paired-statistical per-regime gate ----
    report = {"dataset": args.dataset, "seeds": seeds, "gap_floor": GAP_FLOOR,
              "n_mixtures": len(grid), "regimes": {}}
    passed = 0
    for regime in regimes:
        means = {L: float(np.mean(store[regime][L]["aulc"])) for L in store[regime]}
        oracle = max(means, key=means.get)
        singles = {L: m for L, m in means.items() if kind[L] == "single"}
        best_single = max(singles, key=singles.get) if singles else None
        uni = means.get("uniform", float("nan"))
        ref = best_single if (best_single and means[best_single] >= (uni if np.isfinite(uni) else -1)) else "uniform"
        gaps = [store[regime][oracle]["aulc"][i] - store[regime][ref]["aulc"][i]
                for i in range(len(seeds))] if ref in store[regime] else []
        mgap, se, ci = mean_ci(gaps) if gaps else (float("nan"), float("nan"), float("nan"))
        lower = mgap - ci if np.isfinite(ci) else float("nan")
        gate = bool(kind.get(oracle) == "mix" and np.isfinite(mgap) and mgap >= GAP_FLOOR
                    and (len(seeds) < 3 or (np.isfinite(lower) and lower > 0)))
        passed += int(gate)
        report["regimes"][regime] = {
            "oracle": oracle, "oracle_aulc": means[oracle],
            "oracle_weights": next(g["w"] for g in grid if g["label"] == oracle),
            "ref": ref, "best_single": best_single,
            "mean_gap": mgap, "gap_se": se, "gap_ci95_lower": lower,
            "gate_pass": gate,
            "oracle_aulc_mid": float(np.mean(store[regime][oracle]["aulc_mid"])),
            "oracle_aulc_high": float(np.mean(store[regime][oracle]["aulc_high"])),
            "means": means,
        }
        print(f"[GATE {regime}] oracle={oracle}({means[oracle]:.4f}) ref={ref} "
              f"gap={mgap:+.4f} ci95_low={lower:+.4f} -> {'PASS' if gate else 'fail'}", flush=True)

    report["regimes_passed"] = passed
    report["regimes_total"] = len(regimes)
    report["dataset_gate"] = bool(passed >= (len(regimes) + 1) // 2)   # majority

    out_dir = os.path.join(ROOT, cfg.eval.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = "smoke_" if args.smoke else ""
    with open(os.path.join(out_dir, f"b1_oracle_{tag}{args.dataset}.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, f"b1_wstar_{tag}{args.dataset}.json"), "w") as f:
        json.dump({r: report["regimes"][r]["oracle_weights"] for r in regimes}, f, indent=2)

    print(f"\n[B1 SUMMARY {args.dataset}] regimes_passed={passed}/{len(regimes)} "
          f"dataset_gate={'PASS' if report['dataset_gate'] else 'FAIL'}", flush=True)
    open(os.path.join(out_dir, f"B1_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
