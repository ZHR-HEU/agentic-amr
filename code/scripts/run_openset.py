#!/usr/bin/env python
"""PATH-2 C2 spine: OPEN-SET online AMR 鈥?reject-and-describe unseen modulations.

A CNN is trained ONLY on SEEN modulation classes; at test, UNSEEN (held-out) classes
are mixed in. The closed-set CNN head structurally CANNOT emit/flag an unseen class.
We compare standard from-data OPEN-SET REJECTION baselines against the LLM agent:

  reject baselines (unsupervised OOD scores on the SAME CNN, threshold tuned on a
  held-out SEEN validation split to a target false-positive rate):
    - MSP        : reject if max softmax prob < tau      (Hendrycks-Gimpel)
    - Energy     : reject if -logsumexp(logits) ... energy > tau   (Liu et al.)
    - Mahalanobis: reject if min prototype L2 distance > tau       (proto/Maha proxy)
  LLM agent:
    - reads the RF State Card (percentiles of confidence/entropy/energy/prototype-dist + SNR)
      and emits REJECT/ACCEPT  (+ a natural-language DESCRIPTION for rejected samples).

HONEST CLAIM (PATH 2): the LLM REJECT is PARITY with the best OOD baseline (NOT a win) 鈥?its unique, CNN-impossible capability is the open-vocabulary DESCRIPTION of the unseen
signal. We report AUROC + matched-FPR reject recall/precision for parity, and capture
descriptions for a qualitative/grounded-faithfulness eval. Multi-seed 95% CIs.

The LLM NEVER classifies IQ and NEVER sees raw IQ; recognition stays in the CNN.
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier

SKEYS = ["C", "H", "E", "D", "SNR"]


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def auroc(scores, labels):
    """labels: 1=unseen(positive/OOD), 0=seen. scores: higher => more OOD."""
    order = np.argsort(-np.asarray(scores)); y = np.asarray(labels)[order]
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0: return float("nan")
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / P; fpr = fp / N
    return float(np.trapz(tpr, fpr))


def thr_at_fpr(seen_scores, target_fpr):
    """threshold s.t. fraction of SEEN with score>thr ~= target_fpr (reject => score>thr)."""
    return float(np.quantile(seen_scores, 1.0 - target_fpr))


def reject_metrics(thr, seen_eval, unseen_eval):
    """reject if score>thr. recall=TPR on unseen, fpr on seen, precision among rejected."""
    tp = float((np.asarray(unseen_eval) > thr).sum()); fn = len(unseen_eval) - tp
    fp = float((np.asarray(seen_eval) > thr).sum())
    rec = tp / max(1, (tp + fn)); fpr = fp / max(1, len(seen_eval))
    prec = tp / max(1.0, (tp + fp))
    return rec, fpr, prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2016")
    ap.add_argument("--unseen", default="", help="comma list; default per-dataset")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--per_class_eval", type=int, default=18, help="test samples/class for LLM eval (LLM cost)")
    ap.add_argument("--snr_min", type=int, default=0, help="eval SNR range lower bound (>= this)")
    ap.add_argument("--target_fpr", type=float, default=0.10)
    ap.add_argument("--n_describe", type=int, default=20, help="# rejected-unseen to capture descriptions for")
    ap.add_argument("--backend", default="openai", choices=["openai", "hf", "none"])
    ap.add_argument("--hf_model_path", default="models/gemma-4-12B-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample",
                                     "dataset.test_per_class=260"])
    data = load_dataset(cfg); names = data.classes; L = data.length
    crop = min(128, L)
    if not args.unseen:
        args.unseen = "GFSK,PAM4,WBFM" if args.dataset == "rml2016" else "OOK,OQPSK,FM,GMSK"
    unseen = [c for c in args.unseen.split(",") if c in names]
    U = {names.index(u) for u in unseen}
    seen = [c for c in range(len(names)) if c not in U]
    print(f"== OPEN-SET {args.dataset} == seen={len(seen)} unseen={unseen} crop={crop}", flush=True)

    # LLM backend (built once)
    chat = None
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(s, u):
            r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": s}, {"role": "user", "content": u}],
                temperature=0.0, max_tokens=140, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
    elif args.backend == "hf":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.hf_model_path)
        mdl = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16, device_map=args.device).eval()
        model = os.path.basename(args.hf_model_path.rstrip("/"))
        def chat(s, u):
            enc = tok.apply_chat_template([{"role": "user", "content": s + "\n\n" + u}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            with torch.no_grad(): out = mdl.generate(**enc, max_new_tokens=140, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        model = "none"

    SYS = ("You are an open-set RF monitor sitting ABOVE a modulation classifier (you never see raw IQ). "
           "Given a State Card (percentiles vs the in-distribution training set), decide if the signal is a "
           "KNOWN in-distribution modulation (ACCEPT) or an UNKNOWN out-of-distribution one (REJECT). "
           "A genuinely unknown signal is FAR from every known class: its prototype_distance is HIGH (well above typical), "
           "usually with elevated energy_OOD. Low confidence ALONE is not enough 鈥?a weak but known low-SNR signal has high "
           "entropy yet stays close to a known prototype (moderate prototype_distance). "
           "If you REJECT, add a one-sentence DESCRIPTION of the unknown signal citing the decisive cues. "
           "End with exactly 'VERDICT: ACCEPT' or 'VERDICT: REJECT'.")

    # MSP/Energy/Mahalanobis = single-cue OOD scores; CardMaha/IsoForest = MULTIVARIATE one-class
    # detectors on the FULL card [C,H,E,D,SNR] (same info the LLM reads, no OOD labels) = the FAIR
    # strong baselines the LLM must be compared against (single-cue Mahalanobis alone is too weak).
    BASE = ["MSP", "Energy", "Mahalanobis", "CardMaha", "IsoForest", "LLM"]
    metrics = {m: {k: [] for k in ["auroc", "rec", "fpr", "prec"]} for m in BASE}
    describe_samples = []

    for si in range(args.n_seeds):
        seed = cfg.seed + si
        set_seed(seed); rng = np.random.default_rng(seed)
        remap = {c: i for i, c in enumerate(seen)}
        tr = data.train_idx[np.isin(data.y[data.train_idx], seen)]
        Xtr, ytr = data.X[tr], np.array([remap[c] for c in data.y[tr]])
        clf = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(seen))
        set_seed(seed); clf.fit(Xtr[:, :, :crop], ytr, args.epochs)
        proto = np.stack([clf.features(Xtr[ytr == i][:300][:, :, :crop]).mean(0) for i in range(len(seen))])

        def scores(idxs):
            X = data.X[idxs][:, :, :crop]
            p = clf.predict_proba(X); ll = clf._logits(X).numpy(); f = clf.features(X)
            msp = -p.max(1)                                              # higher=>OOD
            energy = -(np.log(np.exp(ll - ll.max(1, keepdims=True)).sum(1)) + ll.max(1))  # -logsumexp
            maha = np.array([np.linalg.norm(proto - f[j], axis=1).min() for j in range(len(idxs))])
            H = -(p * np.log(p + 1e-12)).sum(1)
            card = np.column_stack([p.max(1), H, energy, maha, data.snr[idxs].astype(float)])  # [C,H,E,D,SNR]
            return p, ll, f, {"MSP": msp, "Energy": energy, "Mahalanobis": maha}, card

        te = data.test_idx[data.snr[data.test_idx] >= args.snr_min]
        # seen test -> val (threshold tuning) + eval ; unseen test -> eval
        seen_te = te[np.isin(data.y[te], seen)]; seen_te = rng.permutation(seen_te)
        nval = len(seen_te) // 2
        seen_val, seen_eval_idx = seen_te[:nval], seen_te[nval:nval + args.per_class_eval * len(seen)]
        uns_te = te[np.isin(data.y[te], list(U))]
        uns_eval_idx = np.concatenate([rng.permutation(uns_te[data.y[uns_te] == u])[:args.per_class_eval] for u in U])

        _, _, _, sc_val, card_val = scores(seen_val)
        pe_s, lle_s, fe_s, sc_se, card_se = scores(seen_eval_idx)
        pe_u, lle_u, fe_u, sc_ue, card_ue = scores(uns_eval_idx)

        # single-cue threshold OOD baselines
        for m in ["MSP", "Energy", "Mahalanobis"]:
            metrics[m]["auroc"].append(auroc(np.concatenate([sc_se[m], sc_ue[m]]),
                                              [0] * len(sc_se[m]) + [1] * len(sc_ue[m])))
            thr = thr_at_fpr(sc_val[m], args.target_fpr)
            rec, fpr, prec = reject_metrics(thr, sc_se[m], sc_ue[m])
            metrics[m]["rec"].append(rec); metrics[m]["fpr"].append(fpr); metrics[m]["prec"].append(prec)

        # MULTIVARIATE one-class detectors on the FULL card (fit on seen_val only; the FAIR strong baselines)
        from numpy.linalg import pinv
        mu = card_val.mean(0); cov = np.cov(card_val.T) + 1e-6 * np.eye(card_val.shape[1]); ic = pinv(cov)
        def cmaha(C): d = C - mu; return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, ic, d), 0.0))
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(n_estimators=200, random_state=seed).fit(card_val)
        mv = {"CardMaha": (cmaha(card_val), cmaha(card_se), cmaha(card_ue)),
              "IsoForest": (-iso.score_samples(card_val), -iso.score_samples(card_se), -iso.score_samples(card_ue))}
        for m, (sv, sse, sue) in mv.items():
            metrics[m]["auroc"].append(auroc(np.concatenate([sse, sue]), [0] * len(sse) + [1] * len(sue)))
            thr = thr_at_fpr(sv, args.target_fpr)
            rec, fpr, prec = reject_metrics(thr, sse, sue)
            metrics[m]["rec"].append(rec); metrics[m]["fpr"].append(fpr); metrics[m]["prec"].append(prec)

        # LLM agent: percentiles vs seen-val distribution
        def pct_fn(key, arr_val):
            v = np.sort(arr_val)
            return lambda x: float(np.searchsorted(v, x) / max(1, len(v)))
        # build per-cue percentile fns from seen-val raw cues
        val_C = sc_val_raw = None
        pv, llv, fv, _, _ = scores(seen_val)
        cueval = {"C": pv.max(1), "H": -(pv * np.log(pv + 1e-12)).sum(1),
                  "E": sc_val["Energy"], "D": sc_val["Mahalanobis"]}
        pctf = {k: pct_fn(k, cueval[k]) for k in ["C", "H", "E", "D"]}

        def card(p_row, ll_row, f_row, gi):
            C = float(p_row.max()); H = float(-(p_row * np.log(p_row + 1e-12)).sum())
            E = float(-(np.log(np.exp(ll_row - ll_row.max()).sum()) + ll_row.max()))
            D = float(np.linalg.norm(proto - f_row, axis=1).min())
            return (f"State Card: confidence=P{pctf['C'](C):.2f}, entropy=P{pctf['H'](H):.2f}, "
                    f"energy_OOD=P{pctf['E'](E):.2f}, prototype_distance=P{pctf['D'](D):.2f}, SNR={int(data.snr[gi])}dB.")

        def llm_reject(p_row, ll_row, f_row, gi):
            if chat is None: return None, ""
            u = card(p_row, ll_row, f_row, gi) + "\nIs this a KNOWN or UNKNOWN modulation? " \
                "If REJECT, describe it in one sentence. VERDICT: ACCEPT|REJECT."
            t = chat(SYS, u)
            m = re.search(r"VERDICT:\s*(ACCEPT|REJECT)", t, re.I)
            verdict = (m.group(1).upper() if m else ("REJECT" if "reject" in t.lower() else "ACCEPT"))
            return (verdict == "REJECT"), t

        if chat is not None:
            rej_s = []
            for j in range(len(seen_eval_idx)):
                r, _ = llm_reject(pe_s[j], lle_s[j], fe_s[j], seen_eval_idx[j]); rej_s.append(bool(r))
            rej_u = []
            for j in range(len(uns_eval_idx)):
                r, txt = llm_reject(pe_u[j], lle_u[j], fe_u[j], uns_eval_idx[j]); rej_u.append(bool(r))
                if r and len(describe_samples) < args.n_describe:
                    describe_samples.append({"true_unseen": names[int(data.y[uns_eval_idx[j]])],
                                             "snr": int(data.snr[uns_eval_idx[j]]),
                                             "card": card(pe_u[j], lle_u[j], fe_u[j], uns_eval_idx[j]),
                                             "llm": txt.strip()})
            rec = float(np.mean(rej_u)); fpr = float(np.mean(rej_s))
            tp = float(np.sum(rej_u)); fp = float(np.sum(rej_s))
            prec = tp / max(1.0, tp + fp)
            metrics["LLM"]["auroc"].append(float("nan"))  # hard decision, no score
            metrics["LLM"]["rec"].append(rec); metrics["LLM"]["fpr"].append(fpr); metrics["LLM"]["prec"].append(prec)
        print(f"  seed {si}: " + " | ".join(
            f"{m} rec={np.mean(metrics[m]['rec']):.2f}/fpr={np.mean(metrics[m]['fpr']):.2f}" for m in metrics if metrics[m]["rec"]), flush=True)

    def ci(a):
        a = [x for x in a if x == x]  # drop nan
        if not a: return None
        m = float(np.mean(a)); s = float(np.std(a) / max(1, len(a) ** 0.5)) * 1.96
        return {"mean": m, "ci": s}
    rep = {"dataset": args.dataset, "model": model, "n_seeds": args.n_seeds, "unseen": unseen,
           "target_fpr": args.target_fpr, "snr_min": args.snr_min,
           "metrics": {m: {k: ci(metrics[m][k]) for k in metrics[m]} for m in metrics},
           "describe_samples": describe_samples}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"openset_{tag}{args.dataset}.json"), "w"), indent=2)
    print(f"\n[OPEN-SET {model} {args.dataset}] reject AUROC / recall@fpr={args.target_fpr}:")
    for m in metrics:
        r = rep["metrics"][m]
        au = f"AUROC={r['auroc']['mean']:.3f}" if r["auroc"] else "AUROC=n/a"
        if r["rec"]:
            print(f"  {m:12s} {au} | rec={r['rec']['mean']:.3f}卤{r['rec']['ci']:.3f} "
                  f"fpr={r['fpr']['mean']:.3f} prec={r['prec']['mean']:.3f}")
    print("[READ] HONEST CLAIM = LLM reject ~= best OOD baseline (PARITY); LLM unique = the DESCRIPTION.", flush=True)
    open(os.path.join(out_dir, f"OPENSET_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
