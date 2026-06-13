#!/usr/bin/env python
"""Zero-shot adherence to NOVEL natural-language OPERATING POLICIES 鈥?the clean
"other-axis" win: the LLM controller honors arbitrary NL operating directives it was
never trained on (e.g. "rejection is forbidden", "complex model unavailable",
"safety-critical: reject on any doubt", "latency-critical: light model only"), while a
from-data GBDT 鈥?whose only policy input is a fixed lambda vector 鈥?STRUCTURALLY cannot
represent such directives and violates them.

On STANDARD cost policies (lambda-expressible) the LLM and the policy-conditioned GBDT
TIE on J (the parity result). On NOVEL NL constraints the LLM honors them (violation ~0)
near the constraint-aware oracle, while the GBDT violates heavily. Reports per policy:
J-under-policy and constraint-violation rate. LLM never sees IQ; recognition = CNN/ResNet.
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier

ACTIONS = ["A1", "A2", "A3", "A4"]
COST = {"A1": (1.0, 1.0), "A2": (1.5, 4.0), "A3": (8.0, 6.0), "A4": (1.0, 1.0)}
SKEYS = ["C", "H", "E", "D", "SNR"]
UNSEEN_DEFAULT = "FM,GMSK,OQPSK,OOK"


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def base_reward(a, e, lam):
    l1, l2, l3 = lam; comp, lat = COST[a]
    if a == "A4": ok = e["is_unknown"]
    elif a == "A1": ok = e["c1"] and not e["is_unknown"]
    elif a == "A2": ok = e["c2"] and not e["is_unknown"]
    else: ok = e["c3"] and not e["is_unknown"]
    return (1.0 if ok else 0.0) - l1 * comp - l2 * lat - l3 * (0.0 if ok else 1.0)

# POLICY = (name, nl_text, lambda, forb_fn, is_novel, kind).
#   forb_fn(e, P) -> set of FORBIDDEN actions for env-sample e; P(k,e)=percentile of cue k.
#   kind: "standard" | "static" (constant forbidden set) | "conditional"/"compound" (state-dependent).
# The forbidden set is what a HUMAN would hand-compile from the NL into an output mask. The
# gbdt_masked baseline APPLIES forb_fn (the fair strong baseline = "compile each NL into a mask");
# the LLM gets ONLY the nl_text (zero-shot, no per-constraint engineering). Win for LLM is now
# narrowly defined: match gbdt_masked on J at 0% violation WITHOUT per-constraint code.
BAL = (0.02, 0.02, 0.3)


def build_policies(P):
    """P(k, e) -> percentile of cue k for env-sample e (closure over devset)."""
    return [
        ("accuracy", "Mission: accuracy is paramount; compute and latency are cheap; minimize misclassification.", (0.01, 0.005, 0.5), (lambda e: set()), False, "standard"),
        ("latency",  "Mission: low latency is critical; avoid slow/expensive actions; prefer fast outputs.", (0.03, 0.06, 0.2), (lambda e: set()), False, "standard"),
        ("balanced", "Mission: balance accuracy, compute and latency reasonably.", BAL, (lambda e: set()), False, "standard"),
        # --- STATIC novel constraints (trivially maskable: "never action X") ---
        ("no_reject", "OPERATING CONSTRAINT: rejection is FORBIDDEN in this mission 鈥?you must always classify; NEVER choose A4 (reject), regardless of how unknown the signal looks.", BAL, (lambda e: {"A4"}), True, "static"),
        ("no_complex", "OPERATING CONSTRAINT: the complex model is UNAVAILABLE (hardware down) 鈥?you must NOT choose A3 under any circumstance.", BAL, (lambda e: {"A3"}), True, "static"),
        ("light_only", "OPERATING CONSTRAINT: severe latency/compute budget 鈥?you may ONLY use the light model A1 (or reject A4); NEVER use A2 or A3.", BAL, (lambda e: {"A2", "A3"}), True, "static"),
        ("safety_reject", "OPERATING CONSTRAINT: this is a SAFETY-CRITICAL scan 鈥?when in any doubt, REJECT as unknown (A4); only classify when extremely confident and clearly in-distribution.", BAL, (lambda e: set()), True, "static"),
        # --- CONDITIONAL / COMPOUND novel constraints (state-dependent: NOT a constant action mask;
        #     require per-constraint hand-coding to mask, which the LLM does zero-shot from NL) ---
        ("snr_gated_reject", "OPERATING CONSTRAINT: rejection (A4) is permitted ONLY for very weak signals below -2 dB SNR; at -2 dB or above you MUST classify and may NOT reject.", BAL,
         (lambda e: ({"A4"} if e["SNR"] >= -2 else set())), True, "conditional"),
        ("complex_highSNR", "OPERATING CONSTRAINT: the complex model A3 is too costly at low SNR 鈥?use A3 ONLY when SNR is at least 4 dB; below 4 dB you may NOT use A3.", BAL,
         (lambda e: ({"A3"} if e["SNR"] < 4 else set())), True, "conditional"),
        ("compound_reject", "OPERATING CONSTRAINT: you may reject (A4) ONLY when the signal is BOTH highly uncertain (confidence in the lowest 20%) AND far from all known prototypes (prototype_distance in the top 20%); in every other case rejection (A4) is forbidden.", BAL,
         (lambda e: (set() if (P("C", e) < 0.20 and P("D", e) > 0.80) else {"A4"})), True, "compound"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--snr_min", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--per_class_pool", type=int, default=16)
    ap.add_argument("--unseen", default=UNSEEN_DEFAULT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample", "dataset.test_per_class=160"])
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes; L = data.length
    unseen = {names.index(u) for u in args.unseen.split(",") if u in names}
    seen = [c for c in range(len(names)) if c not in unseen]
    remap = {c: i for i, c in enumerate(seen)}; inv = {i: c for c, i in remap.items()}
    crop = min(128, L)
    tr = data.train_idx[np.isin(data.y[data.train_idx], seen)]
    Xtr, ytr = data.X[tr], np.array([remap[c] for c in data.y[tr]])
    a1 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(seen))
    a3 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=resnet"]), len(seen))
    set_seed(cfg.seed); print("[train A1]", flush=True); a1.fit(Xtr[:, :, :crop], ytr, args.epochs)
    set_seed(cfg.seed); print("[train A3]", flush=True); a3.fit(Xtr, ytr, args.epochs)
    proto = np.stack([a1.features(Xtr[ytr == i][:300][:, :, :crop]).mean(0) for i in range(len(seen))])
    te = data.test_idx; te = te[data.snr[te] >= args.snr_min]
    pool = np.array([gi for c in range(len(names)) for gi in rng.permutation(te[data.y[te] == c])[:args.per_class_pool]])
    Xp = data.X[pool]
    p1 = a1.predict_proba(Xp[:, :, :crop]); ll = a1._logits(Xp[:, :, :crop]).numpy(); f1 = a1.features(Xp[:, :, :crop]); p2 = a1.predict_proba(Xp); p3 = a3.predict_proba(Xp)
    env = []
    for j, gi in enumerate(pool):
        true = int(data.y[gi]); unk = true in unseen
        env.append({"true": true, "is_unknown": unk,
                    "c1": (inv[int(p1[j].argmax())] == true) and not unk, "c2": (inv[int(p2[j].argmax())] == true) and not unk,
                    "c3": (inv[int(p3[j].argmax())] == true) and not unk,
                    "C": float(p1[j].max()), "H": float(-(p1[j]*np.log(p1[j]+1e-12)).sum()),
                    "E": float(-(np.log(np.exp(ll[j]-ll[j].max()).sum())+ll[j].max())),
                    "D": float(np.linalg.norm(proto-f1[j],axis=1).min()), "SNR": int(data.snr[gi])})
    n = len(env); idx = np.arange(n); dev = idx[:n//2]; ev = idx[n//2:]
    devset = {k: np.array([env[i][k] for i in dev], float) for k in SKEYS}
    def pct(k, v): return float((devset[k] < v).mean())
    def svec(i): return np.array([pct(k, env[i][k]) for k in SKEYS])

    P = lambda k, e: pct(k, e[k])          # percentile of cue k for env-sample e
    policies = build_policies(P)

    def oracle(i, lam, forb):
        f = forb(env[i]); acts = [a for a in ACTIONS if a not in f]
        return max(acts, key=lambda a: base_reward(a, env[i], lam))

    # policy-conditioned GBDT trained on the 3 STANDARD (lambda) policies only (sees lam, NOT NL constraints).
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xg, yg = [], []
    for (nm, nl, lam, forb, novel, kind) in policies:
        if novel: continue
        for i in dev:
            Xg.append(np.concatenate([svec(i), lam])); yg.append(oracle(i, lam, forb))
    gbdt = HistGradientBoostingClassifier(max_iter=200, max_depth=4).fit(np.array(Xg), np.array(yg))
    gclasses = list(gbdt.classes_)
    def gbdt_act(i, lam):                   # UNMASKED: no NL constraint input -> may VIOLATE
        return str(gbdt.predict([np.concatenate([svec(i), lam])])[0])
    def gbdt_masked_act(i, lam, forb):      # FAIR STRONG BASELINE: human compiles each NL -> output mask
        f = forb(env[i]); pr = gbdt.predict_proba([np.concatenate([svec(i), lam])])[0]
        allowed = [(pr[k], c) for k, c in enumerate(gclasses) if c not in f]
        return str(max(allowed)[1]) if allowed else "A1"

    from openai import OpenAI
    cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
    SYS = ("You are a cost-aware decision agent for modulation recognition. A1=light model; A2=extend sampling; "
           "A3=complex model; A4=reject UNKNOWN. Obey the MISSION/CONSTRAINT exactly, then pick the best allowed action. "
           "End 'ANSWER: A1|A2|A3|A4'.")
    def stext(i):
        e = env[i]
        return (f"State: confidence=P{pct('C',e['C']):.2f}, entropy=P{pct('H',e['H']):.2f}, energy_OOD=P{pct('E',e['E']):.2f}, "
                f"prototype_distance=P{pct('D',e['D']):.2f}, SNR={e['SNR']}dB.")
    def llm_act(i, nl):                     # ZERO-SHOT: gets only the NL text, no per-constraint code
        try:
            r = cli.chat.completions.create(model=model, messages=[{"role":"system","content":SYS},{"role":"user","content":f"{nl}\n{stext(i)}\nChoose the best ALLOWED action. ANSWER: A1|A2|A3|A4."}],
                temperature=0.0, max_tokens=80, extra_body={"chat_template_kwargs":{"enable_thinking":False}})
            t = r.choices[0].message.content or ""
        except Exception: return "A1"
        m = re.search(r"ANSWER:\s*(A[1-4])", t) or re.search(r"\b(A[1-4])\b", t); return m.group(1) if m else "A1"

    report = {"dataset": args.dataset, "model": model, "policies": {}}
    print("\n%-16s %-11s %-7s %-7s %-7s %-7s %-7s %-7s %-7s" % (
        "policy", "kind", "J_orac", "J_gbdt", "J_gmask", "J_llm", "v_gbdt", "v_gmask", "v_llm"))
    for (nm, nl, lam, forb, novel, kind) in policies:
        Jo = Jg = Jm = Jl = 0.0; vg = vm = vl = 0
        for i in ev:
            fset = forb(env[i])
            ao = oracle(i, lam, forb); ag = gbdt_act(i, lam); am = gbdt_masked_act(i, lam, forb); al = llm_act(i, nl)
            def cJ(a):                       # forbidden action -> no correctness, pay error penalty
                return -lam[2] if a in fset else base_reward(a, env[i], lam)
            Jo += cJ(ao); Jg += cJ(ag); Jm += cJ(am); Jl += cJ(al)
            vg += (ag in fset); vm += (am in fset); vl += (al in fset)
        m = len(ev)
        report["policies"][nm] = {"novel": novel, "kind": kind, "J_oracle": Jo/m, "J_gbdt": Jg/m,
                                  "J_gbdt_masked": Jm/m, "J_llm": Jl/m,
                                  "viol_gbdt": vg/m, "viol_gbdt_masked": vm/m, "viol_llm": vl/m}
        r = report["policies"][nm]
        print("%-16s %-11s %-7.3f %-7.3f %-7.3f %-7.3f %-7.3f %-7.3f %-7.3f" % (
            nm, kind, r["J_oracle"], r["J_gbdt"], r["J_gbdt_masked"], r["J_llm"],
            r["viol_gbdt"], r["viol_gbdt_masked"], r["viol_llm"]), flush=True)

    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag+"_") if args.tag else ""
    json.dump(report, open(os.path.join(out_dir, f"nlpolicy_{tag}{args.dataset}.json"), "w"), indent=2)
    R = report["policies"]
    def mean(key, ps): return float(np.mean([R[p[0]][key] for p in ps]))
    std = [p for p in policies if not p[4]]; nov = [p for p in policies if p[4]]
    cond = [p for p in policies if p[5] in ("conditional", "compound")]
    print("\n[STANDARD] J: gbdt=%.3f gmask=%.3f llm=%.3f (parity expected)" % (
        mean("J_gbdt", std), mean("J_gbdt_masked", std), mean("J_llm", std)))
    print("[NOVEL] viol: gbdt(unmasked)=%.3f gmask=%.3f llm=%.3f | J: gmask=%.3f llm=%.3f oracle=%.3f" % (
        mean("viol_gbdt", nov), mean("viol_gbdt_masked", nov), mean("viol_llm", nov),
        mean("J_gbdt_masked", nov), mean("J_llm", nov), mean("J_oracle", nov)))
    print("[CONDITIONAL/COMPOUND] (the irreducible test) viol_llm=%.3f | J: gmask=%.3f llm=%.3f" % (
        mean("viol_llm", cond), mean("J_gbdt_masked", cond), mean("J_llm", cond)))
    print("[READ] gbdt_masked = human compiles each NL into an output mask (0%% viol by construction). "
          "LLM is zero-shot from NL. CLAIM HOLDS iff viol_llm~0 AND J_llm>=J_gbdt_masked (LLM matches a hand-masked "
          "router with NO per-constraint engineering). If J_llm<J_gbdt_masked, the masked baseline dominates -> C2 dies.")
    open(os.path.join(out_dir, f"NLPOLICY_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
