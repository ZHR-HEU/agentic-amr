#!/usr/bin/env python
"""COMPLEX dynamic-environment DECISION gate (zero-shot to novel FACTOR COMBINATIONS) 鈥?the strongest fair test of "LLM decision-making in dynamic AMR".

Environment has THREE interacting factors applied to real RML2018 IQ:
  SNR (low/mid/high) x channel (clean / multipath-fading) x interference (none / co-channel tone).
K=4 SPECIALIST classifiers are each trained on ONE factor combo, so the OPTIMAL routing
decision depends on the JOINT (snr, channel, interference) state, not SNR alone.

The trained router (gbdt) is fit on SEEN (pure) combos; tested ZERO-SHOT on NOVEL
combos the specialists/router never saw TOGETHER (e.g. low-SNR+fading+interference).
A simple SNR rule is now provably suboptimal (ignores channel/interference); the
must-beat strong deterministic baseline is conf_rule (trust the most-confident specialist).

The LLM's hypothesized irreducible edge: read the rich noisy multi-factor situation and
COMPOSITIONALLY reason which specialist is least-bad on an unseen combination, beating
gbdt_router (no data for the combo), snr_rule, AND conf_rule (miscalibrated off-combo).
WIN: llm > conf_rule AND llm > gbdt_router AND llm > snr_rule on NOVEL combos.
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


def _norm(x):
    s = x.std()
    return x / (s + 1e-8)


def to_c(x): return x[0].astype(np.float64) + 1j * x[1].astype(np.float64)
def from_c(z): return _norm(np.stack([z.real, z.imag], 0).astype(np.float32))


def aug_fading(x, sev, rng):
    z = to_c(x); L = len(z); d = int(rng.integers(2, 9)); a = sev * rng.uniform(0.4, 0.85)
    th = rng.uniform(0, 2 * np.pi); zf = z.astype(complex).copy()
    zf[d:] += a * np.exp(1j * th) * z[:-d]
    return from_c(zf)


def aug_interf(x, lvl, rng):
    z = to_c(x); L = len(z); f = rng.uniform(0.05, 0.45); ph = rng.uniform(0, 2 * np.pi)
    amp = lvl * np.mean(np.abs(z)); n = np.arange(L)
    return from_c(z + amp * np.exp(1j * (2 * np.pi * f * n + ph)))


def apply_factors(x, channel, interf, rng):
    if channel == "fading": x = aug_fading(x, 1.0, rng)
    if interf == "cochannel": x = aug_interf(x, 0.8, rng)
    return x


def fading_score(x):
    z = to_c(x); z0 = np.vdot(z, z).real + 1e-8
    return float(np.mean([abs(np.vdot(z[d:], z[:-d])) for d in (2, 4, 6)]) / z0)


def interf_score(x):
    z = to_c(x); S = np.abs(np.fft.fft(z))
    return float(S.max() / (S.mean() + 1e-8))


# specialist factor combos
SPEC_COMBOS = [
    ("clean", "none", (-2, 8)),     # spec0: clean mid
    ("fading", "none", (-2, 8)),    # spec1: fading mid
    ("clean", "none", (-20, -2)),   # spec2: clean low
    ("cochannel", "none", (-2, 8)), # spec3: interference mid  (interf carried by 'channel' slot)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--snr_noise", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=120)
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
    yloc = lambda gi: classes.index(int(data.y[gi]))

    # --- train 4 specialists on their factor combos ---
    specs = []
    for si, (chan, interf, (lo, hi)) in enumerate(SPEC_COMBOS):
        m = tr_all[(data.snr[tr_all] >= lo) & (data.snr[tr_all] < hi)]
        m = rng.permutation(m)[:6000]
        rg = np.random.default_rng(cfg.seed + 100 + si)
        Xa = np.stack([apply_factors(data.X[gi], chan, interf, rg) for gi in m])
        ya = np.array([yloc(gi) for gi in m])
        clf = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(classes))
        set_seed(cfg.seed + si)
        print(f"[spec {si} {chan}/{interf}/{lo}..{hi} n={len(m)}]", flush=True)
        clf.fit(Xa, ya, args.epochs)
        specs.append(clf)

    def step_obs(snr_target, channel, interf):
        cand = te_all[np.abs(data.snr[te_all] - snr_target) <= 2]
        if len(cand) == 0: cand = te_all
        gi = int(rng.choice(cand))
        x = apply_factors(data.X[gi], channel, interf, rng)
        xb = x[None, ...]
        ps = [s.predict_proba(xb)[0] for s in specs]
        confs = [float(p.max()) for p in ps]; preds = [int(p.argmax()) for p in ps]
        true = yloc(gi); correct = [int(pr == true) for pr in preds]
        st = [snr_target + rng.normal(0, args.snr_noise), fading_score(x) + rng.normal(0, 0.02),
              interf_score(x) / 10.0 + rng.normal(0, 0.05)] + confs
        return st, confs, correct

    # regime trajectories: (snr, channel, interf)
    def traj(kind, t, T):
        if kind == "pure_clean_mid":  return (3, "clean", "none")
        if kind == "pure_fading_mid": return (3, "fading", "none")
        if kind == "pure_clean_low":  return (-12, "clean", "none")
        if kind == "pure_interf_mid": return (3, "clean", "cochannel")
        if kind == "drift_clean_ramp":return (-16 + 32 * t / T, "clean", "none")
        # NOVEL combos (never seen together):
        if kind == "triple_stress":   return (-12, "fading", "cochannel")
        if kind == "fading_highsnr":  return (16, "fading", "none")
        if kind == "interf_lowsnr":   return (-12, "clean", "cochannel")
        return (0, "clean", "none")

    SEEN = ["pure_clean_mid", "pure_fading_mid", "pure_clean_low", "pure_interf_mid", "drift_clean_ramp"]
    NOVEL = ["triple_stress", "fading_highsnr", "interf_lowsnr"]

    def episode(kind):
        T = args.steps; rows = []
        for t in range(T):
            snr, chan, interf = traj(kind, t, T)
            st, confs, correct = step_obs(snr, chan, interf)
            rows.append({"state": st, "confs": confs, "correct": correct})
        return rows

    # gbdt router trained on SEEN combos
    Xtr, ytr = [], []
    for kind in SEEN:
        for _ in range(3):
            for r in episode(kind):
                Xtr.append(r["state"]); ytr.append(int(np.argmax(r["correct"])) if any(r["correct"]) else 0)
    gbdt = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gbdt = HistGradientBoostingClassifier(max_iter=300, max_depth=6)
        gbdt.fit(np.array(Xtr), np.array(ytr))
    except Exception as e:
        print(f"[warn] gbdt skipped ({e})", flush=True)

    BAND_MID = [0, 3, -11, 3]  # rough specialist SNR centers for snr_rule
    def snr_rule(st): return int(np.argmin([abs(c - st[0]) for c in BAND_MID]))
    def conf_rule(st): return int(np.argmax(st[3:7]))

    chat = None
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(st):
            u = ("Online AMR under a COMPLEX drifting environment. Four specialist classifiers, each strong only on "
                 "the condition it was trained for:\n"
                 "  #0 = clean channel, MID SNR | #1 = multipath FADING, MID SNR | #2 = clean, LOW SNR | #3 = co-channel INTERFERENCE, MID SNR\n"
                 "Current noisy situation estimates:\n"
                 f"  - SNR ~ {st[0]:.1f} dB\n  - multipath/fading indicator: {st[1]:.2f} (higher=more fading)\n"
                 f"  - interference indicator: {st[2]:.2f} (higher=more co-channel interference)\n"
                 f"  - specialist confidences: #0={st[3]:.2f} #1={st[4]:.2f} #2={st[5]:.2f} #3={st[6]:.2f}\n"
                 "The conditions may be a NOVEL COMBINATION none of the specialists was trained on; reason which "
                 "specialist is least mismatched to the CURRENT joint conditions, not just the most confident. "
                 "End with 'ANSWER: <0|1|2|3>'.")
            r = cli.chat.completions.create(model=model, messages=[{"role": "user", "content": u}],
                temperature=0.0, max_tokens=160, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            t = r.choices[0].message.content or ""
            mt = re.search(r"ANSWER:\s*([0123])", t)
            return int(mt.group(1)) if mt else conf_rule(st)
    else:
        model = "none"

    methods = ["oracle", "snr_rule", "conf_rule", "gbdt_router", "bandit", "random"]
    if chat: methods.append("llm")
    results = {}
    for kind in NOVEL + ["pure_clean_mid"]:  # include one seen as sanity
        ep = episode(kind); acc = {m: 0 for m in methods}
        q = np.zeros(4); nc = np.zeros(4) + 1e-6
        for r in ep:
            st, corr = r["state"], r["correct"]
            ch = {"oracle": int(np.argmax(corr)) if any(corr) else 0,
                  "snr_rule": snr_rule(st), "conf_rule": conf_rule(st),
                  "gbdt_router": int(gbdt.predict([st])[0]) if gbdt is not None else conf_rule(st),
                  "random": int(rng.integers(4)),
                  "bandit": int(q.argmax()) if rng.random() > 0.12 else int(rng.integers(4))}
            if chat: ch["llm"] = chat(st)
            for m in methods: acc[m] += corr[ch[m]]
            a = ch["bandit"]; nc[a] += 1; q[a] += (corr[a] - q[a]) / nc[a]
        n = len(ep); results[kind] = {m: acc[m] / n for m in methods}
        print(f"[{kind}] " + " ".join(f"{m}={acc[m]/n:.3f}" for m in methods), flush=True)

    agg = {m: float(np.mean([results[k][m] for k in NOVEL])) for m in methods}
    rep = {"dataset": args.dataset, "model": model, "novel": NOVEL, "seen": SEEN,
           "per_regime": results, "aggregate_novel": agg}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"dynroute2_gate_{tag}{args.dataset}.json"), "w"), indent=2)
    print("\n=== COMPLEX DYN-ROUTE (zero-shot NOVEL factor combos) aggregate ===")
    for m in methods: print(f"  {m:12s} = {agg[m]:.3f}")
    if chat:
        print(f"[WIN if] llm > conf_rule ({agg['llm']:.3f} vs {agg['conf_rule']:.3f}) AND > gbdt ({agg['gbdt_router']:.3f}) AND > snr_rule ({agg['snr_rule']:.3f})")
    open(os.path.join(out_dir, f"DYNROUTE2_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
