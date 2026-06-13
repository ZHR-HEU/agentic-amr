#!/usr/bin/env python
"""Full-supervision AMR backbone baseline: train AMRNet (cnn / cnngru) to convergence
on the full dataset and report standard recognition accuracy (overall + per-SNR curve
+ high-SNR), the classifier baseline a paper needs.

Usage:
  python scripts/train_supervised.py --dataset rml2016 --backbone cnn --epochs 60
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier, accuracy


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2016")
    ap.add_argument("--backbone", default="cnn", choices=["cnn", "cnngru", "resnet", "cldnn"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    ov = [f"dataset.name={args.dataset}", f"model.backbone={args.backbone}",
          "dataset.normalize=per_sample", "dataset.test_per_class=200"]
    cfg = load_config(args.config, ov)
    set_seed(cfg.seed)
    t0 = time.time()
    data = load_dataset(cfg)
    Xtr, ytr = data.X[data.train_idx], data.y[data.train_idx]
    Xte, yte = data.X[data.test_idx], data.y[data.test_idx]
    snr_te = data.snr[data.test_idx]
    print(f"== supervised {args.dataset}/{args.backbone} == train={len(ytr)} test={len(yte)} "
          f"classes={data.n_classes} load={time.time()-t0:.0f}s", flush=True)

    clf = Classifier(cfg, data.n_classes)
    best = 0.0
    done = 0
    while done < args.epochs:
        step = min(args.eval_every, args.epochs - done)
        clf.fit(Xtr, ytr, step); done += step
        acc = accuracy(clf.predict_proba(Xte), yte)
        best = max(best, acc)
        print(f"  epoch {done}/{args.epochs} test_acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    proba = clf.predict_proba(Xte); pred = proba.argmax(1)
    overall = float((pred == yte).mean())
    per_snr = {}
    for s in sorted(np.unique(snr_te).tolist()):
        m = snr_te == s
        per_snr[int(s)] = float((pred[m] == yte[m]).mean())
    hi = snr_te >= 10
    high = float((pred[hi] == yte[hi]).mean()) if hi.any() else float("nan")
    rep = {"dataset": args.dataset, "backbone": args.backbone, "epochs": args.epochs,
           "n_train": int(len(ytr)), "n_test": int(len(yte)), "n_classes": data.n_classes,
           "overall_acc": overall, "best_overall_acc": best, "high_snr_acc(>=10dB)": high,
           "per_snr_acc": per_snr, "minutes": round((time.time() - t0) / 60, 1)}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    with open(os.path.join(out_dir, f"supervised_{tag}{args.dataset}_{args.backbone}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\n[SUPERVISED {args.dataset}/{args.backbone}] overall={overall:.4f} "
          f"high_snr(>=10dB)={high:.4f} best={best:.4f} ({rep['minutes']}min)", flush=True)
    print("[per-SNR] " + ", ".join(f"{s}:{a:.2f}" for s, a in per_snr.items()), flush=True)
    open(os.path.join(out_dir, f"SUP_DONE_{tag}{args.dataset}_{args.backbone}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
