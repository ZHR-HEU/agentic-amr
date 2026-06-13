"""Compute bootstrap 95% CI for Novel F1 across all intent result files."""
import json, glob, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
results = sorted(glob.glob(str(ROOT / "results" / "agent_intent_*.json")))

def f1_set(gt, pred):
    gt, pred = set(gt), set(pred)
    if not gt and not pred:
        return 1.0
    tp = len(gt & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

B = 8000
rng = np.random.default_rng(42)

for fp in results:
    d = json.load(open(fp))
    tag = os.path.basename(fp)
    model = d.get("model", tag)

    novel = [rec for rec in d["dump"] if rec["tier"] == "novel"]
    if not novel or "llm" not in novel[0]:
        continue

    scores = np.array([f1_set(rec["gt"], rec["llm"]) for rec in novel])
    mean = float(scores.mean())
    n = len(scores)

    boot = np.array([scores[rng.integers(0, n, size=n)].mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    base = [rec for rec in d["dump"] if rec["tier"] == "base"]
    base_scores = np.array([f1_set(rec["gt"], rec["llm"]) for rec in base]) if base else np.array([])
    base_mean = float(base_scores.mean()) if len(base_scores) > 0 else None

    print(f"{model:35s}  Base={base_mean:.3f}  Novel={mean:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]  n={n}")

    # also bootstrap gap vs best router
    rtr_scores = {}
    for method in ["router_nn", "router_char", "router_ml", "router_ml_char", "router_desc"]:
        rs = np.array([f1_set(rec["gt"], rec.get(method, [])) for rec in novel])
        rtr_scores[method] = rs

    best_rtr_name = max(rtr_scores, key=lambda k: rtr_scores[k].mean())
    best_rtr = rtr_scores[best_rtr_name]
    gap = scores - best_rtr
    boot_gap = np.array([gap[rng.integers(0, n, size=n)].mean() for _ in range(B)])
    g_lo, g_hi = np.percentile(boot_gap, [2.5, 97.5])
    print(f"  gap vs {best_rtr_name}: {gap.mean():.3f} [{g_lo:.3f}, {g_hi:.3f}]")
    print()
