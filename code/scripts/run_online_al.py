#!/usr/bin/env python
"""Run one online-AL episode with the configured controller/regime.

Usage:
  python scripts/run_online_al.py --set controller.name=llm stream.regime=snr_ramp
  python scripts/run_online_al.py --set controller.name=fixed_hybrid dataset.name=rml2016
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
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--set", dest="overrides", nargs="*", default=[],
                    help="dotted overrides, e.g. controller.name=llm stream.regime=snr_step")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    print(f"== run_online_al == dataset={cfg.dataset.name} regime={cfg.stream.regime} "
          f"controller={cfg.controller.name} seed={cfg.seed}")
    data = load_dataset(cfg)
    print(f"[data] X={data.X.shape} classes={data.n_classes} "
          f"train={len(data.train_idx)} test={len(data.test_idx)}")

    ctrl = build_controller(cfg.controller.name, cfg, rng)
    out = run_episode(cfg, data, ctrl, rng, verbose=True)
    print(f"[summary] {json.dumps(out['summary'])}")

    out_dir = os.path.join(ROOT, cfg.eval.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fname = args.out or (f"{cfg.controller.name}_{cfg.dataset.name}_"
                         f"{cfg.stream.regime}_seed{cfg.seed}.json")
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
