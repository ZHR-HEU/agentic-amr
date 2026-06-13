#!/usr/bin/env python
"""B0 acceptance test: run a SHORT online-AL episode end-to-end for `random` and
`llm` controllers on RML2016, exercising the vLLM endpoint.

Usage:
  python scripts/sanity_check.py --dataset rml2016 --controllers random,llm
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
sys.path.insert(0, ROOT)

import numpy as np
import torch

from amrl.config import load_config
from amrl.data import load_dataset
from amrl.controllers import build_controller
from amrl.episode import run_episode


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2016")
    ap.add_argument("--controllers", default="random,llm")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    # short, fast episode for sanity
    overrides = [
        f"dataset.name={args.dataset}",
        "stream.n_steps=6", "stream.pool_size=300",
        "stream.budget_per_step=15", "stream.seed_labeled=150",
        "model.warmup_epochs=5", "model.update_epochs=2",
    ]
    cfg = load_config(args.config, overrides)
    print(f"== amrl sanity_check == dataset={cfg.dataset.name} "
          f"controllers={args.controllers} device={cfg.device}")

    t0 = time.time()
    data = load_dataset(cfg)
    print(f"[data] loaded {cfg.dataset.name}: X={data.X.shape} classes={data.n_classes} "
          f"train={len(data.train_idx)} test={len(data.test_idx)} ({time.time()-t0:.1f}s)")

    results = {}
    ok = True
    for name in [c.strip() for c in args.controllers.split(",") if c.strip()]:
        print(f"\n----- controller: {name} -----")
        set_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        try:
            ctrl = build_controller(name, cfg, rng)
            out = run_episode(cfg, data, ctrl, rng, verbose=True)
            s = out["summary"]
            print(f"[summary:{name}] {json.dumps(s)}")
            # acceptance checks
            finite = np.isfinite(s["final_acc"]) and np.isfinite(s["aulc"])
            if name == "llm":
                calls = s.get("llm_calls", 0)
                fb = s.get("llm_fallbacks", 0)
                print(f"[llm] calls={calls} fallbacks={fb}")
                if calls == 0 or fb == calls:
                    print(f"[FAIL] llm controller never produced a parsed weight vector")
                    ok = False
            if not finite:
                print(f"[FAIL] non-finite metrics for {name}")
                ok = False
            results[name] = s
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] controller {name} crashed: {type(e).__name__}: {e}")
            ok = False
            results[name] = {"error": f"{type(e).__name__}: {e}"}

    out_dir = os.path.join(ROOT, cfg.eval.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sanity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}  ({time.time()-t0:.1f}s total)")
    print("SANITY_OK" if ok else "SANITY_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
