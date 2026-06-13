#!/usr/bin/env python
"""RICHER-INPUT attempt (PATH-2, user-directed): can an LLM produce a SUBSTANTIVE,
objectively-correct open-vocabulary DESCRIPTION of an UNSEEN modulation from interpretable
signal FEATURES (not the abstract CNN state card)? The closed-set CNN structurally cannot
describe (it emits a wrong seen-class index); a from-data rule/matcher CAN map features->
properties, so the LLM must MATCH-OR-BEAT those fair baselines while doing it zero-shot.

Input to the LLM: the 12 textbook descriptors (rf_features) + their physical-meaning LEGEND
+ SNR. Output: a one-line description + a STRUCTURED tail 'PROPS: envelope=...; modtype=...'
parsed and scored OBJECTIVELY against the true modulation's known physical properties.

Baselines producing the SAME property judgments (fair):
  - rule     : hand-thresholds on rf_features, thresholds CALIBRATED on SEEN classes
  - kbmatch  : nearest SEEN class by feature distance -> that class's known properties
Metric: per-property accuracy (envelope, modtype) on HELD-OUT unseen classes. Multi-seed.

HONEST: this directly retries the angle that failed as zero-shot NAMING (chance 0.091);
the bet is that coarse PROPERTY description is more directly readable from features. If the
LLM only ties the rule, the description is real-but-rule-matchable (report honestly).
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.rf_features import compute_descriptors, FEATURE_NAMES, FEATURE_LEGEND

# Ground-truth physical properties of the modulations we describe (textbook).
# envelope in {constant, varying}; modtype in {amplitude, phase, frequency}.
PROPS = {
    # RML2016 names
    "GFSK": ("constant", "frequency"), "PAM4": ("varying", "amplitude"), "WBFM": ("constant", "frequency"),
    "CPFSK": ("constant", "frequency"), "GMSK": ("constant", "frequency"), "BPSK": ("constant", "phase"),
    "QPSK": ("constant", "phase"), "8PSK": ("constant", "phase"), "QAM16": ("varying", "amplitude"),
    "QAM64": ("varying", "amplitude"), "AM-DSB": ("varying", "amplitude"), "AM-SSB": ("varying", "amplitude"),
    # RML2018 names
    "OOK": ("varying", "amplitude"), "4ASK": ("varying", "amplitude"), "8ASK": ("varying", "amplitude"),
    "16PSK": ("constant", "phase"), "32PSK": ("constant", "phase"), "OQPSK": ("constant", "phase"),
    "16QAM": ("varying", "amplitude"), "32QAM": ("varying", "amplitude"), "64QAM": ("varying", "amplitude"),
    "128QAM": ("varying", "amplitude"), "256QAM": ("varying", "amplitude"), "16APSK": ("varying", "amplitude"),
    "32APSK": ("varying", "amplitude"), "64APSK": ("varying", "amplitude"), "128APSK": ("varying", "amplitude"),
    "FM": ("constant", "frequency"), "AM-DSB-WC": ("varying", "amplitude"), "AM-DSB-SC": ("varying", "amplitude"),
    "AM-SSB-WC": ("varying", "amplitude"), "AM-SSB-SC": ("varying", "amplitude"),
}
ENV = ["constant", "varying"]; MOD = ["amplitude", "phase", "frequency"]


def parse_props(t):
    env = mod = None
    m = re.search(r"envelope\s*=\s*(constant|varying)", t, re.I)
    if m: env = m.group(1).lower()
    m = re.search(r"modtype\s*=\s*(amplitude|phase|frequency)", t, re.I)
    if m: mod = m.group(1).lower()
    if env is None:  # fallback: scan prose
        if re.search(r"constant[\s-]*envelope|constant amplitude", t, re.I): env = "constant"
        elif re.search(r"amplitude[\s-]*var|varying amplitude|amplitude[- ]bearing", t, re.I): env = "varying"
    if mod is None:
        if re.search(r"frequency[\s-]*mod|freq[\s-]*shift|FSK|FM\b|CPM", t, re.I): mod = "frequency"
        elif re.search(r"phase[\s-]*shift|PSK", t, re.I): mod = "phase"
        elif re.search(r"amplitude[\s-]*shift|ASK|QAM|PAM|amplitude mod", t, re.I): mod = "amplitude"
    return env, mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2016")
    ap.add_argument("--unseen", default="")
    ap.add_argument("--snr_min", type=int, default=6, help="describe at usable SNR (feature quality)")
    ap.add_argument("--per_class", type=int, default=24)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--backend", default="openai", choices=["openai", "hf", "none"])
    ap.add_argument("--hf_model_path", default="models/gemma-4-12B-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample",
                                     "dataset.test_per_class=300"])
    data = load_dataset(cfg); names = data.classes
    if not args.unseen:
        args.unseen = "GFSK,PAM4,WBFM" if args.dataset == "rml2016" else "OOK,OQPSK,FM,GMSK"
    unseen = [u for u in args.unseen.split(",") if u in names and u in PROPS]
    seen = [c for c in names if c not in unseen and c in PROPS]
    print(f"== DESCRIBE {args.dataset} == unseen={unseen} (held-out classes to describe) snr>={args.snr_min}", flush=True)

    # LLM backend
    chat = None
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(s, u):
            r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": s}, {"role": "user", "content": u}],
                temperature=0.0, max_tokens=160, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
    elif args.backend == "hf":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.hf_model_path)
        mdl = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16, device_map=args.device).eval()
        model = os.path.basename(args.hf_model_path.rstrip("/"))
        def chat(s, u):
            enc = tok.apply_chat_template([{"role": "user", "content": s + "\n\n" + u}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            with torch.no_grad(): out = mdl.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        model = "none"

    SYS = ("You are an RF signal analyst. From interpretable signal descriptors (you never see raw IQ), describe the "
           "signal's PHYSICAL modulation properties. " + FEATURE_LEGEND +
           "\nState whether the envelope is constant or varying, and whether the dominant modulation mechanism is "
           "amplitude, phase, or frequency. One short sentence, then EXACTLY: "
           "'PROPS: envelope=<constant|varying>; modtype=<amplitude|phase|frequency>'.")

    def feat_text(f):
        return "Descriptors: " + ", ".join(f"{FEATURE_NAMES[i]}={f[i]:.3g}" for i in range(len(FEATURE_NAMES)))

    methods = ["rule", "kbmatch"] + (["llm"] if chat else [])
    acc = {m: {"env": [], "mod": [], "both": []} for m in methods}
    samples = []

    for si in range(args.n_seeds):
        rng = np.random.default_rng(cfg.seed + si)
        te = data.test_idx[data.snr[data.test_idx] >= args.snr_min]
        # SEEN feature means + calibrated rule thresholds
        seen_feats = {}
        for c in seen:
            idx = rng.permutation(te[data.y[te] == names.index(c)])[:120]
            if len(idx) == 0: continue
            seen_feats[c] = compute_descriptors(data.X[idx]).mean(0)
        seen_mat = np.array(list(seen_feats.values())); seen_lbl = list(seen_feats.keys())
        # rule thresholds from seen: amp_std (idx0) splits envelope; ifreq_std (idx3) flags frequency-mod
        amp_vals = np.array([seen_feats[c][0] for c in seen_lbl])
        env_true = np.array([1 if PROPS[c][0] == "varying" else 0 for c in seen_lbl])
        # threshold = midpoint between mean amp_std of constant vs varying seen classes
        ta = (amp_vals[env_true == 1].mean() + amp_vals[env_true == 0].mean()) / 2 if (env_true == 1).any() and (env_true == 0).any() else np.median(amp_vals)
        ifreq_vals = np.array([seen_feats[c][3] for c in seen_lbl])
        freq_true = np.array([1 if PROPS[c][1] == "frequency" else 0 for c in seen_lbl])
        tf = (ifreq_vals[freq_true == 1].mean() + ifreq_vals[freq_true == 0].mean()) / 2 if (freq_true == 1).any() and (freq_true == 0).any() else np.percentile(ifreq_vals, 75)

        def rule_pred(f):
            env = "varying" if f[0] > ta else "constant"
            if f[3] > tf: mod = "frequency"
            elif f[0] > ta: mod = "amplitude"
            else: mod = "phase"
            return env, mod

        def kb_pred(f):
            d = np.linalg.norm(seen_mat - f, axis=1); c = seen_lbl[int(d.argmin())]
            return PROPS[c][0], PROPS[c][1]

        for c in unseen:
            idx = rng.permutation(te[data.y[te] == names.index(c)])[:args.per_class]
            F = compute_descriptors(data.X[idx])
            te_env, te_mod = PROPS[c]
            for j in range(len(idx)):
                f = F[j]
                preds = {"rule": rule_pred(f), "kbmatch": kb_pred(f)}
                if chat:
                    txt = chat(SYS, feat_text(f) + f", SNR={int(data.snr[idx[j]])}dB.\nDescribe properties. PROPS: envelope=..; modtype=..")
                    preds["llm"] = parse_props(txt)
                    if len(samples) < 24:
                        samples.append({"true": c, "true_props": [te_env, te_mod], "snr": int(data.snr[idx[j]]),
                                        "feat": {FEATURE_NAMES[k]: round(float(f[k]), 3) for k in range(len(FEATURE_NAMES))},
                                        "llm": " ".join(txt.split())})
                for m in methods:
                    pe, pm = preds[m]
                    acc[m]["env"].append(1.0 if pe == te_env else 0.0)
                    acc[m]["mod"].append(1.0 if pm == te_mod else 0.0)
                    acc[m]["both"].append(1.0 if (pe == te_env and pm == te_mod) else 0.0)
        print(f"  seed {si}: " + " | ".join(f"{m} both={np.mean(acc[m]['both']):.2f}" for m in methods), flush=True)

    rep = {"dataset": args.dataset, "model": model, "n_seeds": args.n_seeds, "unseen": unseen,
           "metrics": {m: {k: {"mean": float(np.mean(acc[m][k])), "n": len(acc[m][k])} for k in acc[m]} for m in methods},
           "samples": samples}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"describe_{tag}{args.dataset}.json"), "w"), indent=2)
    print(f"\n[DESCRIBE {model} {args.dataset}] property accuracy on UNSEEN classes:")
    for m in methods:
        r = rep["metrics"][m]
        print(f"  {m:8s} envelope={r['env']['mean']:.3f} modtype={r['mod']['mean']:.3f} both={r['both']['mean']:.3f}")
    print("[READ] WIN if llm >= rule/kbmatch on property accuracy (zero-shot, no calibration). "
          "If llm ~ rule, description is real but rule-matchable (DPI) -> report honestly.", flush=True)
    open(os.path.join(out_dir, f"DESCRIBE_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
