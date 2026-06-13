#!/usr/bin/env python
"""Dynamic-scenario ADAPTATION for the interpretable agent: a NON-STATIONARY stream with
drift (SNR degradation) + concept drift (an UNKNOWN modulation appears mid-stream) +
recovery. Tests whether ONLINE experience memory lets the agent ADAPT, vs an offline
deterministic controller that cannot, vs an online bandit that can.

Honest framing (not an accuracy-win): expected `llm_online_mem ~ online_bandit > static_gbdt`
under drift, and `online_mem > nomem` (memory enables adaptation) 鈥?the dynamic-scenario
evidence for the memory + adaptive pillars; LLM additionally emits drift-aware rationales.

Reuses the interp decision env (A1 light / A2 extend / A3 complex / A4 reject-unknown).
Stream segments: [0,S) high-SNR known | [S,2S) SNR ramps down | [2S,3S) low-SNR + UNKNOWN
injected | [3S,4S) recovery. Reports per-segment J, novelty-reject recall, adaptation lag.
"""
import argparse, json, os, re, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier

UNSEEN_DEFAULT = "FM,GMSK,OQPSK,OOK"
ACTIONS = ["A1", "A2", "A3", "A4"]
COST = {"A1": (1.0, 1.0), "A2": (1.5, 4.0), "A3": (8.0, 6.0), "A4": (1.0, 1.0)}
LAM = (0.02, 0.02, 0.3)  # balanced
SKEYS = ["C", "H", "E", "D", "SNR"]
ACT_CUE = {"A1": ["confiden", "clear", "easy", "reliable", "in-distribution", "in distribution"],
           "A2": ["snr", "noise", "weak", "extend", "longer", "low signal", "dropping", "degrad"],
           "A3": ["complex", "hard", "confus", "uncertain", "ambig", "finer", "high-order"],
           "A4": ["unknown", "ood", "out-of", "out of", "distance", "prototype", "novel", "reject", "unfamiliar", "energy"]}


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def reward(a, e):
    l1, l2, l3 = LAM; comp, lat = COST[a]
    if a == "A4": correct = e["is_unknown"]
    elif a == "A1": correct = e["c1"] and not e["is_unknown"]
    elif a == "A2": correct = e["c2"] and not e["is_unknown"]
    else: correct = e["c3"] and not e["is_unknown"]
    return (1.0 if correct else 0.0) - l1 * comp - l2 * lat - l3 * (0.0 if correct else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seg", type=int, default=60)         # steps per segment (x4)
    ap.add_argument("--win", type=int, default=40)         # online memory window
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--unseen", default=UNSEEN_DEFAULT)
    ap.add_argument("--backend", default="openai", choices=["openai", "hf", "none"])
    ap.add_argument("--hf_model_path", default="models/gemma-4-12B-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--crop", type=int, default=0, help="short-window length for A1/A2 (0=min(128,L))")
    ap.add_argument("--seed", type=int, default=-1, help="override config seed for multi-seed CIs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    _ov = [f"dataset.name={args.dataset}", "dataset.normalize=per_sample", "dataset.test_per_class=200"]
    if args.seed >= 0: _ov.append(f"seed={args.seed}")
    cfg = load_config(args.config, _ov)
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes; L = data.length
    unseen_ids = {names.index(u) for u in args.unseen.split(",") if u in names}
    seen_ids = [c for c in range(len(names)) if c not in unseen_ids]
    remap = {c: i for i, c in enumerate(seen_ids)}; inv = {i: c for c, i in remap.items()}
    crop = args.crop if args.crop > 0 else min(128, L)
    print(f"== interp DYNAMIC == backend={args.backend} crop={crop}/{L} seen={len(seen_ids)} unseen={sorted(unseen_ids)}", flush=True)

    tr = data.train_idx[np.isin(data.y[data.train_idx], seen_ids)]
    Xtr, ytr = data.X[tr], np.array([remap[c] for c in data.y[tr]])
    a1 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(seen_ids))
    a3 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=resnet"]), len(seen_ids))
    set_seed(cfg.seed); print("[train A1]", flush=True); a1.fit(Xtr[:, :, :crop], ytr, args.epochs)
    set_seed(cfg.seed); print("[train A3]", flush=True); a3.fit(Xtr, ytr, args.epochs)
    proto = np.stack([a1.features(Xtr[ytr == i][:300][:, :, :crop]).mean(0) for i in range(len(seen_ids))])

    te = data.test_idx
    def sample_idx(snr_target, unknown):
        ids = unseen_ids if unknown else set(seen_ids)
        cand = te[np.isin(data.y[te], list(ids)) & (np.abs(data.snr[te] - snr_target) <= 2)]
        if len(cand) == 0: cand = te[np.isin(data.y[te], list(ids))]
        return int(rng.choice(cand))

    def make_env(gi):
        x = data.X[gi:gi + 1]; true = int(data.y[gi]); unk = true in unseen_ids
        p1 = a1.predict_proba(x[:, :, :crop])[0]; lg = a1._logits(x[:, :, :crop]).numpy()[0]; f1 = a1.features(x[:, :, :crop])[0]
        p2 = a1.predict_proba(x)[0]; p3 = a3.predict_proba(x)[0]
        return {"true": true, "is_unknown": unk,
                "c1": (inv[int(p1.argmax())] == true) and not unk, "c2": (inv[int(p2.argmax())] == true) and not unk,
                "c3": (inv[int(p3.argmax())] == true) and not unk,
                "C": float(p1.max()), "H": float(-(p1 * np.log(p1 + 1e-12)).sum()),
                "E": float(-(np.log(np.exp(lg - lg.max()).sum()) + lg.max())),
                "D": float(np.linalg.norm(proto - f1, axis=1).min()), "SNR": int(data.snr[gi])}

    # calibration pool (percentiles) from seen, broad SNR
    calib = []
    for c in seen_ids:
        for gi in rng.permutation(te[data.y[te] == c])[:12]:
            calib.append(make_env(int(gi)))
    calset = {k: np.array([e[k] for e in calib], float) for k in SKEYS}
    def pct(k, v): return float((calset[k] < v).mean())
    def svec(e): return np.array([pct(k, e[k]) for k in SKEYS])

    # offline gbdt trained on calib (oracle actions) 鈥?does NOT adapt online
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xg = [svec(e) for e in calib]; yg = [max(ACTIONS, key=lambda a: reward(a, e)) for e in calib]
    static_gbdt = HistGradientBoostingClassifier(max_iter=200, max_depth=4).fit(np.array(Xg), np.array(yg))

    # FIXED threshold rule (tuned on calib, anchored on the original distribution) 鈥?the
    # strongest deterministic baseline: it can reject high-OOD novelties INSTANTLY by threshold.
    def tune_rule():
        best, bJ = None, -1e9
        for ct in [0.4, 0.5, 0.6, 0.7]:
            for ot in [0.8, 0.85, 0.9]:
                for esc in ["A3", "A2"]:
                    def f(e, ct=ct, ot=ot, esc=esc):
                        if pct("D", e["D"]) > ot or pct("E", e["E"]) > ot: return "A4"
                        if pct("C", e["C"]) > ct: return "A1"
                        return esc
                    J = np.mean([reward(f(e), e) for e in calib])
                    if J > bJ: bJ, best = J, f
        return best
    fixed_rule = tune_rule()
    # ONLINE sliding-window rule: recomputes percentiles from the recent window (adapts thresholds)
    def online_rule(e, win):
        if len(win) < 8: return fixed_rule(e)
        def opc(k, v): a = np.array([w[k] for w in win]); return float((a < v).mean())
        if opc("D", e["D"]) > 0.85 or opc("E", e["E"]) > 0.85: return "A4"
        if opc("C", e["C"]) > 0.6: return "A1"
        return "A3"

    # build the non-stationary stream
    S = args.seg; T = 4 * S
    stream = []
    for t in range(T):
        if t < S: snr, unk_p = rng.uniform(12, 18), 0.0
        elif t < 2 * S: snr, unk_p = 16 - 16 * (t - S) / S, 0.0          # ramp down
        elif t < 3 * S: snr, unk_p = rng.uniform(-4, 4), 0.5             # low SNR + novelty
        else: snr, unk_p = rng.uniform(10, 16), 0.0                       # recovery
        unknown = rng.random() < unk_p
        stream.append(make_env(sample_idx(snr, unknown)))
    seg_of = lambda t: t // S

    SYS = ("You are an EXPLAINABLE, ADAPTIVE cost-aware decision agent for online modulation recognition in a "
           "NON-STATIONARY environment (SNR drifts; new/unknown signals can appear). A1=light model (confident, "
           "in-distribution); A2=extend sampling (LOW SNR); A3=complex model (hard/uncertain known); A4=reject "
           "UNKNOWN (high energy_OOD/prototype_distance). An unknown can look confident: if OOD/distance is high, "
           "prefer A4 regardless of confidence. Use RECENT EXPERIENCES to adapt to the current regime. One short "
           "sentence citing the decisive cue (mention drift/trend if relevant), then 'ANSWER: A1|A2|A3|A4'.")
    def build_user(e, memrows, recent_snrs):
        mt = ""
        if memrows:
            mt = "\nRecent experiences (state-percentiles -> good action):\n" + "\n".join(
                f"  conf P{m[0][0]:.2f} ent P{m[0][1]:.2f} energy P{m[0][2]:.2f} dist P{m[0][3]:.2f} snr {m[1]}dB -> {m[2]}" for m in memrows)
        trend = ""
        if len(recent_snrs) >= 4:
            trend = f"\nRecent SNR trend (oldest->newest): {', '.join(f'{s:.0f}' for s in recent_snrs[-6:])} dB."
        return (f"State: confidence=P{pct('C',e['C']):.2f}, entropy=P{pct('H',e['H']):.2f}, energy_OOD=P{pct('E',e['E']):.2f}, "
                f"prototype_distance=P{pct('D',e['D']):.2f}, SNR={e['SNR']}dB.{trend}{mt}\nReason then choose. ANSWER: A1|A2|A3|A4.")
    def parse_act(txt):
        m = re.search(r"ANSWER:\s*(A[1-4])", txt) or re.search(r"\b(A[1-4])\b", txt)
        return m.group(1) if m else "A1"

    chat = None
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(e, memrows, recent_snrs):
            try:
                r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": build_user(e, memrows, recent_snrs)}],
                    temperature=0.0, max_tokens=170, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                txt = r.choices[0].message.content or ""
            except Exception:
                return "A1", ""
            return parse_act(txt), txt
    elif args.backend == "hf":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.hf_model_path)
        mdl = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16, device_map=args.device).eval()
        model = os.path.basename(args.hf_model_path.rstrip("/"))
        def chat(e, memrows, recent_snrs):
            enc = tok.apply_chat_template([{"role": "user", "content": SYS + "\n\n" + build_user(e, memrows, recent_snrs)}],
                                          add_generation_prompt=True, return_tensors="pt", return_dict=True)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=170, do_sample=False, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            return parse_act(txt), txt
    else:
        model = "none"

    base_methods = ["oracle", "static_gbdt", "fixed_rule", "online_rule", "bandit"]
    methods = base_methods + (["llm_nomem", "llm_online_mem"] if chat else [])
    perseg = {m: collections.defaultdict(lambda: [0.0, 0]) for m in methods}  # m -> seg -> [sumJ,n]
    nov_rej = {m: [0, 0] for m in methods}     # novelty segment: rejected unknowns / total unknowns
    rej_trace = {m: [] for m in methods}       # per-step reject(0/1) for adaptation lag
    onlinemem = collections.deque(maxlen=args.win)   # (svec, snr, best_action) revealed online
    onwin = collections.deque(maxlen=args.win)       # raw env dicts for online_rule percentiles
    q = np.zeros(4); nc = np.zeros(4) + 1e-6; recent_snrs = []; faith = [0, 0]; examples = []
    for t, e in enumerate(stream):
        seg = seg_of(t); sv = svec(e); recent_snrs.append(e["SNR"])
        best = max(ACTIONS, key=lambda a: reward(a, e))
        ch = {"oracle": best, "static_gbdt": str(static_gbdt.predict([sv])[0]),
              "fixed_rule": fixed_rule(e), "online_rule": online_rule(e, onwin),
              "bandit": (ACTIONS[int(q.argmax())] if rng.random() > 0.15 else ACTIONS[int(rng.integers(4))])}
        onwin.append(e)
        if chat:
            mrows = []
            if onlinemem:
                qd = np.argsort([np.linalg.norm(m[0] - sv) for m in onlinemem])[:args.topk]
                mrows = [onlinemem[k] for k in qd]
            an, _ = chat(e, [], recent_snrs); ch["llm_nomem"] = an
            am, tm = chat(e, mrows, recent_snrs); ch["llm_online_mem"] = am
            if e["is_unknown"] or t % 12 == 0:
                if len(examples) < 16:
                    examples.append({"t": t, "seg": seg, "SNR": e["SNR"], "unk": e["is_unknown"],
                                     "ood_P": round(pct("E", e["E"]), 2), "dist_P": round(pct("D", e["D"]), 2),
                                     "action": am, "rationale": tm.strip().replace("\n", " ")[:200]})
            cue = any(kw in tm.lower() for kw in ACT_CUE.get(am, []))
            grounded = ((pct("C", e["C"]) > .55 and pct("E", e["E"]) < .7 and pct("D", e["D"]) < .7) if am == "A1"
                        else e["SNR"] <= 5 if am == "A2"
                        else (pct("H", e["H"]) > .55 or pct("C", e["C"]) < .5) if am == "A3"
                        else (pct("E", e["E"]) > .65 or pct("D", e["D"]) > .65))
            faith[1] += 1; faith[0] += (cue and grounded)
        ba = ACTIONS.index(ch["bandit"]); rwd = reward(ch["bandit"], e); nc[ba] += 1; q[ba] += (rwd - q[ba]) / nc[ba]
        onlinemem.append((sv, e["SNR"], best))
        for m in methods:
            perseg[m][seg][0] += reward(ch[m], e); perseg[m][seg][1] += 1
            rej_trace[m].append(1 if ch[m] == "A4" else 0)
            if seg == 2 and e["is_unknown"]:
                nov_rej[m][1] += 1; nov_rej[m][0] += (ch[m] == "A4")
        if (t + 1) % 60 == 0: print(f"  ...{t+1}/{T}", flush=True)

    def lag(m):  # steps after novelty onset (t=2S) until reject-rate(window5)>0.5
        on = 2 * S
        for t in range(on, 3 * S):
            w = rej_trace[m][t:t + 5]
            if len(w) >= 3 and np.mean(w) > 0.5: return t - on
        return args.seg
    rep = {"dataset": args.dataset, "model": model, "T": T, "seg": S,
           "segJ": {m: {s: (perseg[m][s][0] / perseg[m][s][1] if perseg[m][s][1] else 0.0) for s in range(4)} for m in methods},
           "overallJ": {m: float(np.mean([reward(None, None) if False else 0])) for m in methods},
           "novelty_reject_recall": {m: (nov_rej[m][0] / nov_rej[m][1] if nov_rej[m][1] else 0.0) for m in methods},
           "adaptation_lag_steps": {m: lag(m) for m in methods},
           "llm_faithfulness": (faith[0] / faith[1] if faith[1] else None), "examples": examples}
    # overall J
    rep["overallJ"] = {m: float(np.mean([perseg[m][s][0] for s in range(4)]) / max(1, np.mean([perseg[m][s][1] for s in range(4)]))) for m in methods}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"interp_dynamic_{tag}{args.dataset}.json"), "w"), indent=2)
    print("\n=== DYNAMIC ADAPTATION (per-segment J: 0=normal 1=ramp-down 2=lowSNR+UNKNOWN 3=recovery) ===")
    for m in methods:
        sj = rep["segJ"][m]
        print(f"  {m:16s} seg0={sj[0]:.3f} seg1={sj[1]:.3f} seg2(drift+novel)={sj[2]:.3f} seg3={sj[3]:.3f} | "
              f"novReject={rep['novelty_reject_recall'][m]:.3f} adaptLag={rep['adaptation_lag_steps'][m]}")
    if chat: print(f"[interp] llm_online_mem faithfulness={rep['llm_faithfulness']:.3f}")
    print("[honest] expect llm_online_mem ~ bandit > static_gbdt in seg2; online_mem > nomem (memory enables adaptation)")
    open(os.path.join(out_dir, f"INTERPDYN_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
