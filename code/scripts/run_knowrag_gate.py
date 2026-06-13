#!/usr/bin/env python
"""Knowledge-RAG open-set naming gate (tests the user's hypothesis: the recognition
failures are MISSING DOMAIN KNOWLEDGE, fixable by feeding a textual DESCRIPTION +
RAG over a modulation-signature knowledge base 鈥?instead of raw numbers / baked-in knowledge).

For held-out UNKNOWN classes (no labeled examples), name among the unknown candidates using:
  - llm_rag : textual signal description + KB signatures of candidates -> LLM reasons -> name
  - kb_match: deterministic nearest-KB-template matcher (knowledge but NO LLM reasoning) -- the FAIR baseline
  - chance  : 1/|unknown|
WIN = llm_rag > chance AND > prior no-knowledge (~0.29) AND >= kb_match (LLM reasoning adds over rigid matching).
Backends: openai (Qwen3-8B) | hf (Gemma-4-12B).
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.rf_features import compute_descriptors, FEATURE_NAMES
from amrl.mod_knowledge import KB_TEXT, KB_TMPL

UNSEEN_DEFAULT = "256QAM,FM,GMSK,8PSK"
TMPL_DIM = 6  # [env_var, freq_mod, phase_states, amp_levels, c40, analog]


def lvl(p): return "very-low" if p < .15 else "low" if p < .4 else "medium" if p < .6 else "high" if p < .85 else "very-high"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--snr_min", type=int, default=18)
    ap.add_argument("--per_class", type=int, default=10)
    ap.add_argument("--ref_per_class", type=int, default=30)
    ap.add_argument("--unseen", default=UNSEEN_DEFAULT)
    ap.add_argument("--backend", default="openai", choices=["openai", "hf"])
    ap.add_argument("--hf_model_path", default="models/gemma-4-12B-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=none"])
    np.random.seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes
    unseen = [u for u in args.unseen.split(",") if u in names]
    print(f"== knowRAG gate == backend={args.backend} unseen={unseen} snr>={args.snr_min}", flush=True)

    hi = np.where(data.snr >= args.snr_min)[0]
    test_idx, ref_idx = [], []
    for c in range(len(names)):
        idx = rng.permutation(hi[data.y[hi] == c])
        ref_idx += list(idx[:args.ref_per_class])
        if names[c] in unseen:
            test_idx += list(idx[args.ref_per_class:args.ref_per_class + args.per_class])
    Dref = compute_descriptors(data.X[np.array(ref_idx)])
    Dte = compute_descriptors(data.X[np.array(test_idx)]); yte = [data.y[i] for i in test_idx]

    def pct(j, v): return float((Dref[:, j] < v).mean())
    fi = {nm: j for j, nm in enumerate(FEATURE_NAMES)}

    def sigvec(v):  # map to KB_TMPL dims
        return np.array([pct(fi["amp_std"], v[fi["amp_std"]]), pct(fi["ifreq_std"], v[fi["ifreq_std"]]),
                         pct(fi["phase_resid_std"], v[fi["phase_resid_std"]]), pct(fi["amp_std"], v[fi["amp_std"]]),
                         pct(fi["|C40|"], v[fi["|C40|"]]), pct(fi["spec_flatness"], v[fi["spec_flatness"]])])

    def descr(v):
        a, f, p, c, s = (pct(fi["amp_std"], v[fi["amp_std"]]), pct(fi["ifreq_std"], v[fi["ifreq_std"]]),
                         pct(fi["phase_resid_std"], v[fi["phase_resid_std"]]), pct(fi["|C40|"], v[fi["|C40|"]]),
                         pct(fi["spec_flatness"], v[fi["spec_flatness"]]))
        env = "constant (no amplitude info)" if a < 0.4 else "amplitude-varying"
        return (f"Signal description: envelope={env} (amplitude-variation {lvl(a)}); frequency-modulation={lvl(f)}; "
                f"phase-state spread={lvl(p)} (more=>higher PSK order); |C40|={lvl(c)} (high=>low-order PSK, ~0=>QAM); "
                f"spectrum flatness={lvl(s)} (broadband vs tonal).")

    # deterministic KB-matcher baseline (knowledge, no LLM)
    def kb_match(v):
        sv = sigvec(v)
        return min(unseen, key=lambda m: np.linalg.norm(sv - np.array(KB_TMPL[m])))

    # LLM-RAG
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(sysp, usr):
            r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": sysp}, {"role": "user", "content": usr}],
                temperature=0.0, max_tokens=300, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.hf_model_path)
        mdl = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16, device_map=args.device).eval()
        model = os.path.basename(args.hf_model_path)
        def chat(sysp, usr):
            enc = tok.apply_chat_template([{"role": "user", "content": sysp + "\n\n" + usr}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    SYS = ("You are an expert RF analyst. You are given a SIGNAL DESCRIPTION and a KNOWLEDGE BASE of candidate "
           "modulation signatures. Match the description to the signature it best fits, reasoning about envelope, "
           "frequency modulation, phase/amplitude structure and cumulants. End with 'ANSWER: <one candidate>'.")

    def llm_rag(v):
        kb = "\n".join(f"- {m}: {KB_TEXT[m]}" for m in unseen)
        usr = f"{descr(v)}\n\nKnowledge base (candidate modulation signatures):\n{kb}\n\nWhich candidate is it? ANSWER: <one of: {', '.join(unseen)}>."
        txt = chat(SYS, usr)
        m = re.search(r"ANSWER:\s*([A-Za-z0-9\-]+)", txt)
        cand = m.group(1) if m else txt
        for u in unseen:
            if u.lower() == cand.lower(): return u
        for u in unseen:
            if u.lower() in txt.lower(): return u
        return None

    rag_correct, kb_correct, n_unp = 0, 0, 0
    for k in range(len(test_idx)):
        v = Dte[k]; true = names[yte[k]]
        if kb_match(v) == true: kb_correct += 1
        pr = llm_rag(v)
        if pr is None: n_unp += 1
        elif pr == true: rag_correct += 1
        if (k + 1) % 15 == 0: print(f"  ...{k+1}/{len(test_idx)}", flush=True)
    n = len(test_idx)
    rep = {"dataset": args.dataset, "backend": args.backend, "model": model, "unseen": unseen, "n": n,
           "chance": 1.0 / len(unseen), "llm_rag_acc": rag_correct / n, "kb_match_acc": kb_correct / n,
           "n_unparsed": n_unp, "prior_noknowledge_openset": 0.29}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"knowrag_gate_{tag}{args.dataset}.json"), "w"), indent=2)
    print(f"\n[knowRAG {model}] llm_rag={rep['llm_rag_acc']:.3f} kb_match={rep['kb_match_acc']:.3f} "
          f"chance={rep['chance']:.3f} (prior no-knowledge ~0.29) n={n} unparsed={n_unp}", flush=True)
    print("[WIN if] llm_rag > chance AND > 0.29 AND >= kb_match", flush=True)
    open(os.path.join(out_dir, f"KNOWRAG_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
