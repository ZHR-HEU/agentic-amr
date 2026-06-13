#!/usr/bin/env python
"""Dynamic-environment DECISION gate, ZERO-SHOT to UNANTICIPATED drift regimes 鈥?the
strongest untested form of "LLM makes decisions in dynamic AMR".

Setup (RML2018): K=3 SNR-band SPECIALIST classifiers (low / mid / high SNR), each
strong in-band and weak out-of-band. A stream drifts through SNR/channel regimes.
At each step the agent observes a noisy STATE and ROUTES to one specialist (the
decision). Routing the right specialist >> wrong one => real headroom.

The key twist vs gates 1/2/6: the trained router (gbdt) is fit on a set of SEEN
drift regimes, but evaluated ZERO-SHOT on NOVEL, unanticipated regimes (abrupt
flips / oscillation / combined drift). The LLM's hypothesized irreducible edge:
generalize the routing decision to novel regimes via reasoning, beating a trained
router (no data for the regime) AND a fixed a-priori SNR rule.

Methods (all see the SAME state): oracle, snr_rule (route by noisy SNR estimate),
gbdt_router (trained on SEEN regimes), bandit (online eps-greedy), llm (Qwen3-8B).
WIN for the LLM-decision thesis: llm > snr_rule AND llm > gbdt_router on NOVEL regimes.
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


BANDS = [(-20, -2), (-2, 8), (8, 30)]  # low / mid / high SNR specialist training bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--snr_noise", type=float, default=5.0)  # noisy SNR estimate (dB std)
    ap.add_argument("--steps_per_regime", type=int, default=120)
    ap.add_argument("--backend", default="openai", choices=["openai", "none"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample",
                                     "dataset.test_per_class=300"])
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes
    classes = list(range(min(args.n_classes, len(names))))
    keep = lambda idx: idx[np.isin(data.y[idx], classes)]
    tr_all, te_all = keep(data.train_idx), keep(data.test_idx)

    # --- train 3 SNR-band specialists ---
    specs = []
    for bi, (lo, hi) in enumerate(BANDS):
        m = tr_all[(data.snr[tr_all] >= lo) & (data.snr[tr_all] < hi)]
        clf = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(classes))
        set_seed(cfg.seed + bi)
        print(f"[train specialist {bi} band {lo}..{hi} n={len(m)}]", flush=True)
        clf.fit(data.X[m], np.array([classes.index(int(c)) for c in data.y[m]]), args.epochs)
        specs.append(clf)

    def specialist_outputs(gi):
        x = data.X[gi:gi + 1]
        ps = [s.predict_proba(x)[0] for s in specs]
        preds = [int(p.argmax()) for p in ps]
        confs = [float(p.max()) for p in ps]
        true = classes.index(int(data.y[gi]))
        correct = [int(pr == true) for pr in preds]
        return preds, confs, correct

    # --- regime generators: SEEN (for gbdt training) vs NOVEL (zero-shot test) ---
    def regime_snr(kind, t, T):
        if kind == "ramp_up":  return -18 + 36 * t / T
        if kind == "ramp_down":return 18 - 36 * t / T
        if kind == "stable_mid": return 2 + 3 * np.sin(t / 8)
        if kind == "slow_wave": return 6 * np.sin(2 * np.pi * t / T)
        if kind == "abrupt_flip":  return (16 if (t // 20) % 2 == 0 else -16)   # NOVEL
        if kind == "fast_oscillate": return 14 * np.sin(2 * np.pi * t / 11)      # NOVEL
        if kind == "ramp_then_crash": return (-18 + 40 * t / T) if t < T * 0.6 else -18  # NOVEL
        return 0.0

    SEEN = ["ramp_up", "ramp_down", "stable_mid", "slow_wave"]
    NOVEL = ["abrupt_flip", "fast_oscillate", "ramp_then_crash"]

    def build_episode(kind):
        """Return list of (state_vec, confs, correct, true_band, gi). State the agent sees:
        [snr_est_noisy, conf0, conf1, conf2, max_pairwise_disagree, drift_signal]."""
        T = args.steps_per_regime; rows = []; prev_snr = None
        for t in range(T):
            snr = regime_snr(kind, t, T)
            band = min(range(len(BANDS)), key=lambda b: abs((BANDS[b][0] + BANDS[b][1]) / 2 - snr))
            # pick a test sample near this SNR
            cand = te_all[np.abs(data.snr[te_all] - snr) <= 2]
            if len(cand) == 0: cand = te_all
            gi = int(rng.choice(cand))
            preds, confs, correct = specialist_outputs(gi)
            snr_est = snr + rng.normal(0, args.snr_noise)
            disagree = float(np.mean([abs(confs[i] - confs[j]) for i in range(3) for j in range(i + 1, 3)]))
            drift = 0.0 if prev_snr is None else abs(snr - prev_snr); prev_snr = snr
            state = [snr_est, confs[0], confs[1], confs[2], disagree, drift]
            rows.append({"state": state, "confs": confs, "correct": correct, "gi": gi,
                         "true_band": band, "snr_true": float(snr)})
        return rows

    # --- training data for gbdt_router from SEEN regimes (state -> best specialist) ---
    Xtr, ytr = [], []
    for kind in SEEN:
        for _ in range(3):
            for r in build_episode(kind):
                Xtr.append(r["state"]); ytr.append(int(np.argmax(r["correct"]) if any(r["correct"]) else r["state"][0] >= 0))
    gbdt = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gbdt = HistGradientBoostingClassifier(max_iter=250, max_depth=5)
        gbdt.fit(np.array(Xtr), np.array(ytr))
    except Exception as e:
        print(f"[warn] gbdt skipped ({e})", flush=True)

    def snr_rule(state):
        snr_est = state[0]
        return min(range(len(BANDS)), key=lambda b: abs((BANDS[b][0] + BANDS[b][1]) / 2 - snr_est))

    # LLM router
    chat = None
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(state):
            u = (f"Online modulation-recognition under drift. Three classifiers: #0 trained for LOW SNR (-20..-2 dB), "
                 f"#1 for MID SNR (-2..8 dB), #2 for HIGH SNR (8..30 dB). Current (noisy) estimates:\n"
                 f"- estimated SNR ~ {state[0]:.1f} dB\n- classifier confidences: #0={state[1]:.2f}, #1={state[2]:.2f}, #2={state[3]:.2f}\n"
                 f"- confidence spread: {state[4]:.2f}\n- drift since last step: {state[5]:.1f} dB\n"
                 "Pick the single classifier most likely to be CORRECT now. Reason briefly, then end 'ANSWER: <0|1|2>'.")
            r = cli.chat.completions.create(model=model, messages=[{"role": "user", "content": u}],
                temperature=0.0, max_tokens=120, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            t = r.choices[0].message.content or ""
            mt = re.search(r"ANSWER:\s*([012])", t)
            return int(mt.group(1)) if mt else (int(np.argmax(state[1:4])))
    else:
        model = "none"

    def bandit_factory():
        q = np.zeros(3); ncnt = np.zeros(3) + 1e-6
        def pick(state, learn=None):
            if learn is None:
                return int(q.argmax()) if rng.random() > 0.1 else int(rng.integers(3))
            a, rwd = learn; ncnt[a] += 1; q[a] += (rwd - q[a]) / ncnt[a]; return None
        return pick

    # --- evaluate on NOVEL regimes (zero-shot) ---
    results = {}
    methods = ["oracle", "snr_rule", "gbdt_router", "bandit", "random"]
    if chat: methods.append("llm")
    for kind in NOVEL:
        ep = build_episode(kind)
        acc = {m: 0 for m in methods}
        bandit = bandit_factory()
        for r in ep:
            st, corr = r["state"], r["correct"]
            choice = {}
            choice["oracle"] = int(np.argmax(corr)) if any(corr) else 0
            choice["snr_rule"] = snr_rule(st)
            choice["gbdt_router"] = int(gbdt.predict([st])[0]) if gbdt is not None else snr_rule(st)
            choice["random"] = int(rng.integers(3))
            ba = bandit(st); choice["bandit"] = ba
            if chat: choice["llm"] = chat(st)
            for m in methods:
                acc[m] += corr[choice[m]]
            bandit(st, learn=(choice["bandit"], corr[choice["bandit"]]))  # update bandit
        n = len(ep)
        results[kind] = {m: acc[m] / n for m in methods}
        print(f"[NOVEL {kind}] " + " ".join(f"{m}={acc[m]/n:.3f}" for m in methods), flush=True)

    # aggregate over novel regimes
    agg = {m: float(np.mean([results[k][m] for k in NOVEL])) for m in methods}
    rep = {"dataset": args.dataset, "model": model, "novel_regimes": NOVEL, "seen_regimes": SEEN,
           "per_regime": results, "aggregate_novel": agg,
           "best_single_specialist_acc": None}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"dynroute_gate_{tag}{args.dataset}.json"), "w"), indent=2)
    print("\n=== DYN-ROUTE (zero-shot novel regimes) aggregate ===")
    for m in methods: print(f"  {m:12s} = {agg[m]:.3f}")
    if chat:
        print(f"[WIN if] llm > snr_rule ({agg['llm']:.3f} vs {agg['snr_rule']:.3f}) AND llm > gbdt_router ({agg['gbdt_router']:.3f})")
    open(os.path.join(out_dir, f"DYNROUTE_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
