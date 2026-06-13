#!/usr/bin/env python
"""Memory + reasoning agent gate for direction C (open-set / GZSL AMR).

Casts AMR as a state-aware agent DECISION: given RF descriptors + an EXPERIENCE
MEMORY of labeled exemplars, the LLM REASONS to (a) identify a known modulation or
(b) recognize a modulation it has NO examples for (open-set / generalized zero-shot,
where the full label space is known but only some classes have exemplars).

Backends: openai (vLLM endpoint, e.g. Qwen3-8B) | hf (local transformers, e.g.
Gemma-4-12B). Isolates the memory+reasoning effect, then tests model scale.

Baselines: chance ; 1-NN over memory (seen-only -> 0 on unseen by construction).
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from amrl.config import load_config
from amrl.data import load_dataset
from amrl.rf_features import compute_descriptors, FEATURE_LEGEND, FEATURE_NAMES

UNSEEN_DEFAULT = "QAM64,WBFM,GFSK,PAM4"

SYS = ("You are an expert RF/communications signal analyst performing open-set modulation "
       "recognition. You have LABELED EXAMPLES for some modulations (your experience memory). "
       "For OTHER modulations you have NO examples and know them only by name and by how that "
       "family manifests in signal descriptors. A query may belong to EITHER group. Procedure: "
       "(1) compare the query's descriptors to your examples; (2) if it is clearly consistent "
       "with an example's modulation, choose that; (3) if it is INCONSISTENT with all your "
       "example modulations (e.g. amplitude-varying when all examples are constant-envelope, or "
       "different cumulant/spectral signature), it is most likely one of the example-less "
       "modulations 鈥?infer which one from your physics knowledge. Reason briefly, then end with "
       "a line 'ANSWER: <one candidate>'.")


def render(vec):
    return "  ".join(f"{nm}={vec[j]:.3g}" for j, nm in enumerate(FEATURE_NAMES))


class OpenAIChat:
    def __init__(self, endpoint, model):
        from openai import OpenAI
        self.c = OpenAI(base_url=endpoint, api_key="EMPTY")
        self.model = model

    def __call__(self, sys_p, user_p, max_tokens=420):
        try:
            r = self.c.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
                temperature=0.0, max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
        except Exception as e:
            return f"ERR:{type(e).__name__}:{e}"


class HFChat:
    """Local transformers chat (for Gemma-4-12B etc.). Merges system into the user
    turn (Gemma chat templates have no system role)."""
    def __init__(self, model_path, device="cuda:0", load_4bit=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        kw = {"torch_dtype": torch.bfloat16, "device_map": device}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True,
                  bnb_4bit_compute_dtype=torch.bfloat16), "device_map": device}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        self.model.eval()
        self.device = device

    def __call__(self, sys_p, user_p, max_tokens=420):
        msgs = [{"role": "user", "content": sys_p + "\n\n" + user_p}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, in_len:], skip_special_tokens=True)


def parse_answer(txt, names):
    m = re.search(r"ANSWER:\s*([A-Za-z0-9\-]+)", txt)
    if m:
        low = m.group(1).lower()
        for ci, nm in enumerate(names):
            if nm.lower() == low:
                return ci
    for ci, nm in enumerate(names):           # fallback: last mention
        if nm.lower() in txt.lower():
            return ci
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2016")
    ap.add_argument("--snr_min", type=int, default=12)
    ap.add_argument("--per_class", type=int, default=6)
    ap.add_argument("--mem_per_class", type=int, default=40)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--unseen", default=UNSEEN_DEFAULT)
    ap.add_argument("--backend", default="openai", choices=["openai", "hf"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--hf_model_path", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--novelty_gate", action="store_true", help="add deterministic contrastive novelty cue")
    ap.add_argument("--nov_alpha", type=float, default=0.25, help="novelty tolerance (smaller=more aggressive)")
    ap.add_argument("--oracle_name", action="store_true",
                    help="ORACLE-GATE naming: score ONLY unseen-class queries, candidates restricted to "
                         "the unseen names (perfect detection assumed). Tests RF-grounded naming vs 1/|unseen| chance.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=none"])
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg)
    names = data.classes
    unseen = set(x.strip() for x in args.unseen.split(",") if x.strip())
    unseen_ids = {names.index(u) for u in unseen if u in names}
    seen_ids = [c for c in range(len(names)) if c not in unseen_ids]
    seen_names = [names[c] for c in seen_ids]
    unseen_names = [names[c] for c in sorted(unseen_ids)]

    if args.backend == "openai":
        chat = OpenAIChat(args.endpoint or cfg.controller.endpoint, args.model or cfg.controller.model)
        model_label = args.model or cfg.controller.model
    else:
        path = args.hf_model_path or "models/gemma-4-12B-it"
        print(f"[hf] loading {path} on {args.device} (4bit={args.load_4bit}) ...", flush=True)
        chat = HFChat(path, args.device, args.load_4bit)
        model_label = os.path.basename(path)
    print(f"== memRAG gate == backend={args.backend} model={model_label} seen={len(seen_ids)} "
          f"unseen={unseen_names} topk={args.topk}", flush=True)

    hi = np.where(data.snr >= args.snr_min)[0]
    mem_idx, test_idx = [], []
    for c in range(len(names)):
        idxs = rng.permutation(hi[data.y[hi] == c])
        if c in seen_ids:
            mem_idx += list(idxs[:args.mem_per_class])
        test_idx += list(idxs[args.mem_per_class:args.mem_per_class + args.per_class])
    mem_idx = np.array(mem_idx); test_idx = np.array(test_idx)
    Dmem = compute_descriptors(data.X[mem_idx]); ymem = data.y[mem_idx]
    Dtest = compute_descriptors(data.X[test_idx]); ytest = data.y[test_idx]
    mu, sd = Dmem.mean(0), Dmem.std(0) + 1e-9
    Zm, Zt = (Dmem - mu) / sd, (Dtest - mu) / sd
    nn_pred = np.array([ymem[np.argmin(((Zm - z) ** 2).sum(1))] for z in Zt])

    llm_pred = []
    for i in range(len(test_idx)):
        if args.oracle_name and ytest[i] in seen_ids:
            llm_pred.append(-2)            # oracle-naming scores ONLY unseen queries
            continue
        d = ((Zm - Zt[i]) ** 2).sum(1)
        nn = np.argsort(d)[:args.topk]
        ex = "\n".join(f"  example {j+1}: {render(Dmem[k])}  -> modulation: {names[ymem[k]]}"
                       for j, k in enumerate(nn))
        # --- contrastive novelty signal (deterministic): which features fall OUTSIDE the
        # range spanned by the retrieved exemplars -> the query is unlike them (likely novel).
        nov = ""
        if args.novelty_gate:
            ex_lo = Dmem[nn].min(0); ex_hi = Dmem[nn].max(0)
            ex_rng = (ex_hi - ex_lo) + 1e-9
            dev = [(FEATURE_NAMES[j], Dtest[i, j], ex_lo[j], ex_hi[j])
                   for j in range(len(FEATURE_NAMES))
                   if Dtest[i, j] < ex_lo[j] - args.nov_alpha * ex_rng[j] or Dtest[i, j] > ex_hi[j] + args.nov_alpha * ex_rng[j]]
            ex_classes = sorted({names[ymem[k]] for k in nn})
            if dev:
                dl = "; ".join(f"{nm}={v:.3g} (examples spanned {lo:.3g}..{hi:.3g})" for nm, v, lo, hi in dev)
                nov = (f"\nNOVELTY CHECK: your nearest examples are all of class(es) {ex_classes}, but the query "
                       f"is OUTSIDE their range on: {dl}. This means the query is INCONSISTENT with your example "
                       f"classes -> do NOT just copy the nearest example's class; it is likely an example-less "
                       f"modulation whose descriptors match the query (reason from physics).")
            else:
                nov = (f"\nNOVELTY CHECK: the query is within the descriptor range of your nearest examples "
                       f"(class(es) {ex_classes}) -> it is likely one of them.")
        if args.oracle_name:
            cand = unseen_names
            task = (f"\nThis QUERY is a modulation you have NO labeled examples for; it is exactly ONE of: "
                    f"{', '.join(unseen_names)}. Use the query's descriptors, the contrast with your "
                    f"(different-class) examples, and your knowledge of how these modulations manifest, to decide which one.")
        else:
            cand = names
            task = (f"\nModulations you HAVE examples for: {', '.join(seen_names)}."
                    f"\nModulations you have NO examples for (know by name/knowledge only): {', '.join(unseen_names)}."
                    f"\nFull candidate list: {', '.join(names)}\nDecide the modulation (it may be an example-less one).")
        user = (FEATURE_LEGEND + "\nExperience memory (labeled examples, nearest first):\n" + ex + nov + task +
                f"\n\nQUERY signal descriptors:\n  {render(Dtest[i])}\n"
                f"Choose ONE from: {', '.join(cand)}. End with 'ANSWER: <one candidate>'.")
        txt = chat(SYS, user)
        loc = parse_answer(txt, cand)
        llm_pred.append(names.index(cand[loc]) if loc >= 0 else -1)
        if (i + 1) % 15 == 0:
            print(f"  ...{i+1}/{len(test_idx)}", flush=True)
    llm_pred = np.array(llm_pred)

    seen_mask = np.array([y in seen_ids for y in ytest])
    def acc(pred, mask):
        m = mask & (pred != -2)               # -2 = skipped (oracle mode)
        return float((pred[m] == ytest[m]).mean()) if m.any() else float("nan")
    report = {
        "dataset": args.dataset, "backend": args.backend, "model": model_label, "topk": args.topk,
        "oracle_name": bool(args.oracle_name), "nov_alpha": args.nov_alpha if args.novelty_gate else None,
        "n_test": int(len(test_idx)), "chance": 1.0 / len(names),
        "llm_overall": acc(llm_pred, np.ones(len(ytest), bool)),
        "llm_seen": acc(llm_pred, seen_mask), "llm_unseen": acc(llm_pred, ~seen_mask),
        "nn_seen": acc(nn_pred, seen_mask), "nn_unseen": acc(nn_pred, ~seen_mask),
        "n_unparsed": int((llm_pred == -1).sum()), "unseen_classes": unseen_names,
    }
    if args.oracle_name:
        scored = (~seen_mask) & (llm_pred != -2)        # all unseen queries (incl -1 as wrong)
        report["oracle_naming_acc"] = float((llm_pred[scored] == ytest[scored]).mean()) if scored.any() else float("nan")
        report["oracle_naming_chance"] = 1.0 / max(1, len(unseen_names))
        report["n_unseen_scored"] = int(scored.sum())
    out_dir = os.path.join(ROOT, cfg.eval.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    with open(os.path.join(out_dir, f"memrag_gate_{tag}{args.dataset}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[memRAG {args.dataset} | {model_label}] overall={report['llm_overall']:.3f} "
          f"seen={report['llm_seen']:.3f} unseen={report['llm_unseen']:.3f} "
          f"| 1NN seen={report['nn_seen']:.3f} unseen={report['nn_unseen']:.3f} "
          f"| chance={report['chance']:.3f} unparsed={report['n_unparsed']}", flush=True)
    if args.oracle_name:
        print(f"[ORACLE-NAMING {model_label}] unseen_naming_acc={report['oracle_naming_acc']:.3f} "
              f"(fair chance 1/{len(unseen_names)}={report['oracle_naming_chance']:.3f}, "
              f"n={report['n_unseen_scored']}, unseen={unseen_names})", flush=True)
    open(os.path.join(out_dir, f"MEMRAG_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
