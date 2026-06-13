#!/usr/bin/env python
"""LLM-as-feature-program-SYNTHESIZER gate 鈥?the one structurally-different in-domain
shot at an honest LLM win (the LLM emits an ARTIFACT: a symbolic discriminator program,
which a GBDT cannot). For each CNN-confused class pair, several SEARCH POLICIES propose
feature programs over the SAME DSL (amrl/featuredsl.py), scored by the SAME deterministic
verifier (1-D AUC on a val split). Methods: textbook (theory bank), random, enumerate
(depth<=2), gp (genetic), llm (Qwen3-8B). Within an equal eval budget we compare best
separation, search efficiency (evals-to-target), and downstream 2-class TEST accuracy.

PRE-REGISTERED: WIN if llm's best feature >= every search baseline AND >= textbook AND
reaches target AUC with FEWER verifier evals. KILL if enumerate/gp/textbook match the
LLM's best cheaply (=> the modulation-theory prior buys nothing; ~70-90% expected per the
root-cause analysis: the AMR discriminator space is small and theory-mapped).
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier
from amrl.featuredsl import primitives, safe_eval, auc_1d, PRIM_NAMES, TEXTBOOK

OPS = ["+", "-", "*", "/"]; FUNCS = ["abs", "sqrt", "log", "sq"]
GLOSS = ("amp_* = envelope mean/std/kurtosis/skew (multi-amplitude QAM raises amp_kurt/std); "
         "ifreq_std/ifreq_kurt = instantaneous-frequency spread (FM/FSK/GMSK); phase_std = phase-residual "
         "spread (PSK order); C20/C40/C42 = higher-order cumulants normalized by C21 (classic order "
         "discriminators: |C40|,|C42| separate PSK vs QAM and constellation order); spec_flat/spec_peak = "
         "spectral flatness/peakiness; papr = peak-to-average power; env_acf1 = lag-1 envelope autocorrelation.")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def rand_prog(rng):
    p = lambda: rng.choice(PRIM_NAMES); op = lambda: rng.choice(OPS); r = rng.random()
    if r < 0.18: return p()
    if r < 0.36: return f"{rng.choice(FUNCS)}({p()})"
    if r < 0.78: return f"{p()} {op()} {p()}"
    return f"({p()} {op()} {p()}) {op()} {p()}"


def enum_progs(rng):
    progs = list(PRIM_NAMES)
    for f in ["sq", "sqrt", "log"]:
        progs += [f"{f}({a})" for a in PRIM_NAMES]
    for op in ["/", "*", "-"]:
        for i, a in enumerate(PRIM_NAMES):
            for b in PRIM_NAMES[i:]:
                progs.append(f"{a} {op} {b}")
    rng.shuffle(progs); return progs


def to_complex(X):  # X: [N,2,L] -> [N,L] complex
    return X[:, 0, :] + 1j * X[:, 1, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2016")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--snr_lo", type=int, default=0)
    ap.add_argument("--snr_hi", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--llm_props", type=int, default=30)
    ap.add_argument("--target_auc", type=float, default=0.70)
    ap.add_argument("--backend", default="openai", choices=["openai", "none"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample",
                                     "dataset.test_per_class=200", "model.backbone=cnn"])
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes; nC = len(names)

    # train CNN, get confusion at the SNR slice
    cnn = Classifier(cfg, nC); print("[train CNN]", flush=True)
    cnn.fit(data.X[data.train_idx], data.y[data.train_idx], args.epochs)
    sl = data.test_idx[(data.snr[data.test_idx] >= args.snr_lo) & (data.snr[data.test_idx] <= args.snr_hi)]
    pred = cnn.predict_proba(data.X[sl]).argmax(1); true = data.y[sl]
    conf = np.zeros((nC, nC), int)
    for t, p in zip(true, pred):
        if t != p: conf[t, p] += 1
    cm = conf + conf.T
    pairs = []
    for _ in range(args.pairs):
        i, j = np.unravel_index(cm.argmax(), cm.shape)
        if cm[i, j] == 0: break
        pairs.append((int(i), int(j))); cm[i, j] = cm[j, i] = 0
    print(f"[confused pairs] " + ", ".join(f"{names[i]}/{names[j]}" for i, j in pairs), flush=True)

    # raw IQ pools per class at the SNR slice (rebuild from un-normalized? use data.X already normalized)
    def pool(c):
        idx = data.test_idx[(data.y[data.test_idx] == c) & (data.snr[data.test_idx] >= args.snr_lo) & (data.snr[data.test_idx] <= args.snr_hi)]
        return idx
    chat = None
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(a, b, snr):
            u = (f"A CNN confuses modulation {a} vs {b} at SNR around {snr} dB. Propose {args.llm_props} SHORT symbolic "
                 f"discriminator features that, by modulation theory, separate {a} from {b}. Use ONLY these primitives: "
                 f"{', '.join(PRIM_NAMES)}. Meanings: {GLOSS} Allowed operators: + - * / ** and functions abs() sqrt() "
                 f"log() sq(). Output ONE expression per line, nothing else, no '=', no names other than the primitives.")
            r = cli.chat.completions.create(model=model, messages=[{"role": "user", "content": u}],
                temperature=0.3, max_tokens=700, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
    else:
        model = "none"

    def split_eval_test(a, b):
        ia, ib = pool(a), pool(b)
        na, nb = len(ia), len(ib); m = min(na, nb)
        ia = rng.permutation(ia)[:m]; ib = rng.permutation(ib)[:m]
        cut = m // 2
        val_idx = np.concatenate([ia[:cut], ib[:cut]]); te_idx = np.concatenate([ia[cut:], ib[cut:]])
        yval = np.array([0] * cut + [1] * (m - cut))[:len(val_idx)]
        yte = np.array([0] * (m - cut) + [1] * (m - cut))[:len(te_idx)]
        Zval = to_complex(data.X[val_idx]); Zte = to_complex(data.X[te_idx])
        return primitives(Zval), yval, primitives(Zte), yte

    def search(progs_iter, ns_val, yval, budget):
        seen = {}; best = ("", 0.5)
        for prog in progs_iter:
            if len(seen) >= budget: break
            if prog in seen: continue
            try:
                f = safe_eval(prog, ns_val); auc = auc_1d(f, yval)
            except Exception:
                continue
            seen[prog] = auc
            if auc > best[1]: best = (prog, auc)
        # evals-to-target
        et = None; c = 0
        for prog, auc in seen.items():
            c += 1
            if auc >= args.target_auc: et = c; break
        return best, len(seen), et

    def gp_search(ns_val, yval, budget):
        P = 24; pop = [rand_prog(rng) for _ in range(P)]; seen = {}; best = ("", 0.5)
        def ev(pr):
            if pr in seen: return seen[pr]
            try: a = auc_1d(safe_eval(pr, ns_val), yval)
            except Exception: a = 0.5
            seen[pr] = a; return a
        for pr in pop: ev(pr)
        while len(seen) < budget:
            scored = sorted(pop, key=lambda p: seen.get(p, 0.5), reverse=True)
            elite = scored[:max(2, P // 3)]
            child = []
            for _ in range(P):
                if rng.random() < 0.5:
                    a, b = rng.choice(elite), rng.choice(PRIM_NAMES)
                    child.append(f"{a} {rng.choice(OPS)} {b}")
                else:
                    child.append(rand_prog(rng))
            pop = elite + child
            for pr in pop:
                if len(seen) >= budget: break
                ev(pr)
        for pr, a in seen.items():
            if a > best[1]: best = (pr, a)
        et = None; c = 0
        for pr, a in seen.items():
            c += 1
            if a >= args.target_auc: et = c; break
        return best, len(seen), et

    def downstream_acc(prog, ns_val, yval, ns_te, yte):
        try:
            fv = safe_eval(prog, ns_val); ft = safe_eval(prog, ns_te)
        except Exception:
            return 0.5
        thr = np.median(fv); dir_ = 1 if fv[yval == 1].mean() >= fv[yval == 0].mean() else -1
        predv = ((fv - thr) * dir_ > 0).astype(int)
        if (predv == yval).mean() < 0.5: dir_ = -dir_
        predt = ((ft - thr) * dir_ > 0).astype(int)
        return float((predt == yte).mean())

    methods = ["textbook", "random", "enumerate", "gp"] + (["llm"] if chat else [])
    per_pair = {}
    for (a, b) in pairs:
        ns_val, yval, ns_te, yte = split_eval_test(a, b)
        res = {}
        # textbook (fixed bank, eval all)
        res["textbook"] = search(iter(TEXTBOOK), ns_val, yval, len(TEXTBOOK))
        res["random"] = search((rand_prog(rng) for _ in range(args.budget * 3)), ns_val, yval, args.budget)
        res["enumerate"] = search(iter(enum_progs(rng)), ns_val, yval, args.budget)
        res["gp"] = gp_search(ns_val, yval, args.budget)
        if chat:
            props = []
            txt = chat(names[a], names[b], (args.snr_lo + args.snr_hi) // 2)
            for ln in txt.splitlines():
                ln = ln.strip().lstrip("-*0123456789. ").strip("` ")
                if ln and any(p in ln for p in PRIM_NAMES) and "=" not in ln and len(ln) < 80:
                    props.append(ln)
            res["llm"] = search(iter(props), ns_val, yval, args.budget)
        out = {}
        for m in methods:
            (prog, auc), nev, et = res[m]
            acc = downstream_acc(prog, ns_val, yval, ns_te, yte)
            out[m] = {"best_prog": prog, "val_auc": round(auc, 3), "n_eval": nev,
                      "evals_to_target": et, "test_acc": round(acc, 3)}
        per_pair[f"{names[a]}/{names[b]}"] = out
        print(f"[{names[a]}/{names[b]}] " + " | ".join(
            f"{m}:auc={out[m]['val_auc']},acc={out[m]['test_acc']},ev={out[m]['evals_to_target']}" for m in methods), flush=True)

    # aggregate
    agg = {m: {"val_auc": float(np.mean([per_pair[p][m]["val_auc"] for p in per_pair])),
               "test_acc": float(np.mean([per_pair[p][m]["test_acc"] for p in per_pair])),
               "median_evals_to_target": float(np.median([per_pair[p][m]["evals_to_target"] or args.budget for p in per_pair]))}
           for m in methods}
    rep = {"dataset": args.dataset, "model": model, "snr_slice": [args.snr_lo, args.snr_hi],
           "pairs": [f"{names[i]}/{names[j]}" for i, j in pairs], "budget": args.budget,
           "target_auc": args.target_auc, "per_pair": per_pair, "aggregate": agg}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"featuresynth_gate_{tag}{args.dataset}.json"), "w"), indent=2)
    print("\n=== FEATURE-SYNTH aggregate (over confused pairs) ===")
    for m in methods:
        print(f"  {m:10s} val_auc={agg[m]['val_auc']:.3f} test_acc={agg[m]['test_acc']:.3f} med_evals_to_target={agg[m]['median_evals_to_target']:.0f}")
    if chat:
        print(f"[WIN if] llm test_acc & val_auc >= every baseline AND llm med_evals_to_target < random/gp/enumerate")
        print(f"[KILL if] enumerate/gp/textbook match llm's best cheaply")
    open(os.path.join(out_dir, f"FEATURESYNTH_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
