#!/usr/bin/env python
"""REALISTIC TEMPORAL dynamic-environment DECISION gate (vehicular RF, with a clock and
history) 鈥?the strongest, most realistic test of "LLM decision-making in dynamic AMR".

A vehicle drives a timed ROUTE through scenario phases (open_road / highway /
urban_canyon / tunnel / congestion). Each phase drives a continuous-time channel:
  - SNR = phase baseline + AR(1) shadowing + within-phase trend
  - multipath FADING severity (rises with speed/Doppler: highway/urban high, tunnel low)
  - bursty co-channel INTERFERENCE (persists several steps once triggered)
These factors are applied to real RML2018 IQ. The agent observes a NOISY, EMA-SMOOTHED
(hence LAGGING) SNR estimate + fading/interference indicators + the last K steps of
history + elapsed time, and ROUTES to one of 4 specialists (clean-mid / fading-mid /
clean-low / interference-mid). Because the SNR estimate LAGS, during fast transitions
(e.g. entering a tunnel) the instantaneous truth has already moved -> using the temporal
TREND/history to anticipate beats memoryless reaction.

Trained on SEEN routes; tested ZERO-SHOT on NOVEL temporal patterns (tunnel transitions,
sustained congestion, stop-go) absent from training. The LLM's hypothesized irreducible
edge: temporal/anticipatory reasoning over the trajectory narrative + zero-shot to novel
time-patterns, beating gbdt_hist (history features, but no data for the pattern), the
memoryless conf_rule/snr_rule, and an online bandit.
WIN: llm > conf_rule AND llm > gbdt_hist AND llm > snr_rule on NOVEL routes.
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


def _norm(x): return x / (x.std() + 1e-8)
def to_c(x): return x[0].astype(np.float64) + 1j * x[1].astype(np.float64)
def from_c(z): return _norm(np.stack([z.real, z.imag], 0).astype(np.float32))


def aug_fading(x, sev, rng):
    z = to_c(x); d = int(rng.integers(2, 9)); a = sev * rng.uniform(0.4, 0.85); th = rng.uniform(0, 2 * np.pi)
    zf = z.astype(complex).copy(); zf[d:] += a * np.exp(1j * th) * z[:-d]; return from_c(zf)


def aug_interf(x, lvl, rng):
    z = to_c(x); L = len(z); f = rng.uniform(0.05, 0.45); ph = rng.uniform(0, 2 * np.pi)
    amp = lvl * np.mean(np.abs(z)); n = np.arange(L)
    return from_c(z + amp * np.exp(1j * (2 * np.pi * f * n + ph)))


def fading_score(x):
    z = to_c(x); z0 = np.vdot(z, z).real + 1e-8
    return float(np.mean([abs(np.vdot(z[d:], z[:-d])) for d in (2, 4, 6)]) / z0)


def interf_score(x):
    S = np.abs(np.fft.fft(to_c(x))); return float(S.max() / (S.mean() + 1e-8))


SPEC_COMBOS = [("clean", "none", (-2, 8)), ("fading", "none", (-2, 8)),
               ("clean", "none", (-20, -2)), ("cochannel", "none", (-2, 8))]
# phase -> (snr_base, fading_prob, interf_prob)
PHASES = {
    "open_road":    (16, 0.05, 0.05),
    "highway":      (11, 0.85, 0.10),
    "urban_canyon": (2,  0.70, 0.50),
    "tunnel":       (-14, 0.10, 0.00),
    "congestion":   (3,  0.20, 0.75),
}
SEEN_ROUTES = {
    "commute_a": [("open_road", .3), ("highway", .4), ("urban_canyon", .3)],
    "commute_b": [("urban_canyon", .4), ("congestion", .3), ("open_road", .3)],
    "express":   [("highway", .5), ("open_road", .5)],
    "city":      [("urban_canyon", .5), ("congestion", .5)],
}
NOVEL_ROUTES = {  # temporal patterns + tunnel never seen in training
    "into_tunnel":     [("highway", .4), ("tunnel", .3), ("urban_canyon", .3)],
    "congested_canyon":[("congestion", .5), ("urban_canyon", .5)],
    "stop_go":         [("open_road", .2), ("tunnel", .2), ("open_road", .2), ("tunnel", .2), ("open_road", .2)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=130)
    ap.add_argument("--hist", type=int, default=8)
    ap.add_argument("--ema", type=float, default=0.4)
    ap.add_argument("--backend", default="openai", choices=["openai", "none"])
    ap.add_argument("--icl", type=int, default=0, help="# in-context labeled exemplars from SEEN routes (gbdt's training data)")
    ap.add_argument("--feedback", type=int, default=0, help="# recent online (decision->correct?) outcomes shown to the LLM")
    ap.add_argument("--dump", default="", help="if set, dump per-step LLM prompts+truth here for a frontier Codex scorer")
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

    specs = []
    for si, (chan, interf, (lo, hi)) in enumerate(SPEC_COMBOS):
        m = rng.permutation(tr_all[(data.snr[tr_all] >= lo) & (data.snr[tr_all] < hi)])[:6000]
        rg = np.random.default_rng(cfg.seed + 100 + si)
        Xa = np.stack([(aug_fading(data.X[gi], 1.0, rg) if chan == "fading" else
                        aug_interf(data.X[gi], 0.8, rg) if chan == "cochannel" else _norm(data.X[gi].astype(np.float32)))
                       for gi in m])
        clf = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(classes))
        set_seed(cfg.seed + si); print(f"[spec {si} {chan}/{lo}..{hi} n={len(m)}]", flush=True)
        clf.fit(Xa, np.array([yloc(gi) for gi in m]), args.epochs); specs.append(clf)

    def render_step(snr_true, fading_on, interf_on):
        cand = te_all[np.abs(data.snr[te_all] - snr_true) <= 2.5]
        if len(cand) == 0: cand = te_all
        gi = int(rng.choice(cand)); x = _norm(data.X[gi].astype(np.float32))
        if fading_on: x = aug_fading(x, 1.0, rng)
        if interf_on: x = aug_interf(x, 0.8, rng)
        ps = [s.predict_proba(x[None])[0] for s in specs]
        confs = [float(p.max()) for p in ps]
        correct = [int(int(p.argmax()) == yloc(gi)) for p in ps]
        return x, confs, correct

    def simulate(route, seed):
        """Generate a timed episode over a route. Returns per-step dicts with the agent's
        observable history + the true correctness of each specialist."""
        rg = np.random.default_rng(seed); T = args.steps
        # expand route phases to per-step phase list
        segs = []; total = sum(d for _, d in route)
        for ph, d in route:
            segs += [ph] * max(1, int(round(T * d / total)))
        segs = (segs + [route[-1][0]] * T)[:T]
        shadow = 0.0; interf_timer = 0; snr_ema = PHASES[segs[0]][0]
        rows = []
        for t in range(T):
            ph = segs[t]; base, fp, ip = PHASES[ph]
            shadow = 0.85 * shadow + rg.normal(0, 1.6)
            snr_true = base + shadow
            fading_on = rg.random() < fp
            if interf_timer > 0: interf_on = True; interf_timer -= 1
            else:
                interf_on = rg.random() < ip
                if interf_on: interf_timer = int(rg.integers(2, 6))
            x, confs, correct = render_step(snr_true, fading_on, interf_on)
            snr_ema = args.ema * snr_true + (1 - args.ema) * snr_ema  # LAGGING estimate
            obs = {"snr_est": snr_ema + rg.normal(0, 2.0),
                   "fading": fading_score(x) + rg.normal(0, 0.02),
                   "interf": interf_score(x) / 10.0 + rg.normal(0, 0.05),
                   "confs": confs, "t": t, "T": T, "correct": correct, "phase": ph}
            rows.append(obs)
        return rows

    # history feature builder (for gbdt_hist; same info the LLM gets as text)
    def feat(rows, i):
        h = rows[max(0, i - args.hist):i + 1]
        snr = [r["snr_est"] for r in h]; fad = [r["fading"] for r in h]; itf = [r["interf"] for r in h]
        cur = rows[i]
        slope = (snr[-1] - snr[0]) / max(1, len(snr) - 1)
        return [cur["snr_est"], cur["fading"], cur["interf"], *cur["confs"],
                float(np.mean(snr)), slope, float(np.mean(fad)), float(np.mean(itf)),
                float(np.mean([r["interf"] > 1.5 for r in h])), cur["t"] / cur["T"]]

    # train gbdt_hist on SEEN routes
    Xtr, ytr = [], []
    for ri, route in enumerate(SEEN_ROUTES.values()):
        for rep in range(4):
            rows = simulate(route, cfg.seed + 1000 + ri * 7 + rep)
            for i in range(len(rows)):
                Xtr.append(feat(rows, i)); ytr.append(int(np.argmax(rows[i]["correct"])) if any(rows[i]["correct"]) else 0)
    gbdt = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gbdt = HistGradientBoostingClassifier(max_iter=350, max_depth=6).fit(np.array(Xtr), np.array(ytr))
    except Exception as e:
        print(f"[warn] gbdt skipped ({e})", flush=True)

    BAND_MID = [3, 3, -11, 3]
    def snr_rule(rows, i): return 2 if rows[i]["snr_est"] < -4 else int(np.argmax(rows[i]["confs"]))  # SNR-aware: low->clean-low else most-conf
    def conf_rule(rows, i): return int(np.argmax(rows[i]["confs"]))

    # ICL: give the LLM the SAME labeled data the GBDT trained on (in-context few-shot)
    ICL_BLOCK = ""
    if args.icl > 0 and len(Xtr):
        Xa = np.array(Xtr); ya = np.array(ytr); sel = rng.permutation(len(Xa))[:args.icl]
        lines = [f"  SNR~{Xa[k][0]:.0f}dB fading={Xa[k][1]:.2f} interf={Xa[k][2]:.2f} "
                 f"conf[{Xa[k][3]:.2f},{Xa[k][4]:.2f},{Xa[k][5]:.2f},{Xa[k][6]:.2f}] trend={Xa[k][8]:+.1f} "
                 f"-> CORRECT specialist #{int(ya[k])}" for k in sel]
        ICL_BLOCK = ("\nLabeled examples from PAST drives (which specialist turned out correct) -- learn the mapping:\n"
                     + "\n".join(lines) + "\n")

    def build_user(rows, i, fb=""):
        h = rows[max(0, i - args.hist):i + 1]
        log = "\n".join(f"  t={r['t']}: SNR~{r['snr_est']:.0f}dB fading={r['fading']:.2f} interf={r['interf']:.2f} "
                        f"conf[{r['confs'][0]:.2f},{r['confs'][1]:.2f},{r['confs'][2]:.2f},{r['confs'][3]:.2f}]" for r in h)
        return ("Vehicular online AMR. Four specialists: #0 clean/MID-SNR, #1 multipath-FADING/MID-SNR, "
                "#2 clean/LOW-SNR, #3 co-channel-INTERFERENCE/MID-SNR. The SNR estimate LAGS reality, so use the "
                "TREND over time to anticipate transitions." + ICL_BLOCK + fb +
                "\nRecent history (oldest->newest):\n" + log +
                f"\nElapsed: t={rows[i]['t']}/{rows[i]['T']}. Pick the least-mismatched specialist. End 'ANSWER: <0|1|2|3>'.")

    chat = None
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(rows, i, fb=""):
            u = build_user(rows, i, fb)
            r = cli.chat.completions.create(model=model, messages=[{"role": "user", "content": u}],
                temperature=0.0, max_tokens=200, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            t = r.choices[0].message.content or ""; mt = re.search(r"ANSWER:\s*([0123])", t)
            return int(mt.group(1)) if mt else conf_rule(rows, i)
    else:
        model = "none"

    methods = ["oracle", "snr_rule", "conf_rule", "gbdt_hist", "bandit", "random"]
    if chat: methods.append("llm")
    results = {}; dump_records = []
    for name, route in list(NOVEL_ROUTES.items()) + [("commute_a", SEEN_ROUTES["commute_a"])]:
        rows = simulate(route, cfg.seed + 50)
        acc = {m: 0 for m in methods}; q = np.zeros(4); nc = np.zeros(4) + 1e-6; fb_hist = []
        for i in range(len(rows)):
            corr = rows[i]["correct"]
            fbtxt = ("\nYour recent decisions and whether they were correct:\n" + "\n".join(fb_hist[-args.feedback:]) + "\n") if (args.feedback and fb_hist) else ""
            ch = {"oracle": int(np.argmax(corr)) if any(corr) else 0,
                  "snr_rule": snr_rule(rows, i), "conf_rule": conf_rule(rows, i),
                  "gbdt_hist": int(gbdt.predict([feat(rows, i)])[0]) if gbdt is not None else conf_rule(rows, i),
                  "random": int(rng.integers(4)),
                  "bandit": int(q.argmax()) if rng.random() > 0.12 else int(rng.integers(4))}
            if chat: ch["llm"] = chat(rows, i, fbtxt)
            if args.dump:
                dump_records.append({"route": name, "i": i, "prompt": build_user(rows, i, fbtxt), "correct": corr})
            for m in methods: acc[m] += corr[ch[m]]
            a = ch["bandit"]; nc[a] += 1; q[a] += (corr[a] - q[a]) / nc[a]
            if chat:  # online feedback = reward on the CHOSEN action only (same as the bandit; no best-action leak)
                fb_hist.append(f"  chose #{ch['llm']} -> {'CORRECT' if corr[ch['llm']] else 'wrong'}")
        n = len(rows); results[name] = {m: acc[m] / n for m in methods}
        print(f"[{name}] " + " ".join(f"{m}={acc[m]/n:.3f}" for m in methods), flush=True)

    if args.dump:
        json.dump({"records": dump_records, "icl_used": args.icl, "feedback": args.feedback},
                  open(args.dump, "w"), indent=1)
        print(f"[dumped {len(dump_records)} steps -> {args.dump}]", flush=True)
    agg = {m: float(np.mean([results[k][m] for k in NOVEL_ROUTES])) for m in methods}
    rep = {"dataset": args.dataset, "model": model, "icl": args.icl, "feedback": args.feedback,
           "novel_routes": list(NOVEL_ROUTES), "seen_routes": list(SEEN_ROUTES), "per_route": results, "aggregate_novel": agg}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"dynroute3_gate_{tag}{args.dataset}.json"), "w"), indent=2)
    print("\n=== TEMPORAL VEHICULAR DYN-ROUTE (zero-shot NOVEL time-patterns) aggregate ===")
    for m in methods: print(f"  {m:12s} = {agg[m]:.3f}")
    if chat:
        print(f"[WIN if] llm > conf_rule ({agg['llm']:.3f} vs {agg['conf_rule']:.3f}) AND > gbdt_hist ({agg['gbdt_hist']:.3f}) AND > snr_rule ({agg['snr_rule']:.3f})")
    open(os.path.join(out_dir, f"DYNROUTE3_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
