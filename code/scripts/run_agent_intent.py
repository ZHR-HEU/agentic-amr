#!/usr/bin/env python
"""RESULTS-FIRST probe (user-directed): the ONE LLM-irreducible surface left = zero-shot
NL-intent -> tool-PLAN. Numeric conditions stay INSIDE deterministic tools; the LLM only
does SYMBOLIC composition (its strength; NL->formal-artifact = validated win zone), never
numeric evaluation (its proven weakness).

SELF-DEFINED METRIC (not accuracy): tool-set F1 of the agent's plan vs the ground-truth
minimal tool set, reported SEPARATELY on BASE intents (T1/T2, direct) and NOVEL/COMPOUND
intents (T3/T4, held out). Fair baselines that a reviewer WILL build:
  - router_nn  : TF-IDF nearest-neighbor over BASE intents -> copy its tool set (learned router)
  - router_ml  : TF-IDF + per-tool one-vs-rest logistic, trained on BASE (multi-label router)
  - fixed      : always the most-common BASE tool set
WIN (the only honest one): LLM >> routers on NOVEL F1 (routers fail on unseen intents BY
CONSTRUCTION; the LLM generalizes zero-shot) while comparable on BASE. If routers match
NOVEL, the LLM is reducible -> KILL, do not write.
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from amrl.config import load_config

TOOLS = ["check_acc", "detect_drift", "confusion_probe", "openset_reject", "adapt", "allocate_budget"]
TOOL_API = (
    "Available tools (each is deterministic; call by name):\n"
    "- check_acc: report current classifier accuracy (optionally per SNR band).\n"
    "- detect_drift: detect whether/what kind of concept drift is occurring.\n"
    "- confusion_probe: report which modulation classes are most confused.\n"
    "- openset_reject: detect/flag unknown (out-of-distribution) signals.\n"
    "- adapt: update/recalibrate/retrain the classifier.\n"
    "- allocate_budget: decide how to spend the labeling budget across classes.\n")

# (intent, ground-truth minimal tool set, tier). BASE = direct (router-learnable); NOVEL = compound/
# paraphrased/conditional/NL-constrained (held out; not in router training distribution).
BANK = [
    # ---- BASE (T1/T2) ----
    ("What is the current recognition accuracy at low SNR?", {"check_acc"}, "base"),
    ("Report the model's accuracy.", {"check_acc"}, "base"),
    ("Is the data distribution drifting?", {"detect_drift"}, "base"),
    ("Check whether concept drift is happening.", {"detect_drift"}, "base"),
    ("Which modulation classes get confused with each other?", {"confusion_probe"}, "base"),
    ("Show me the top confused class pairs.", {"confusion_probe"}, "base"),
    ("Are there any unknown signals we don't recognize?", {"openset_reject"}, "base"),
    ("Flag out-of-distribution emissions.", {"openset_reject"}, "base"),
    ("Recalibrate the classifier.", {"adapt"}, "base"),
    ("Retrain the model.", {"adapt"}, "base"),
    ("Decide how to allocate the labeling budget.", {"allocate_budget"}, "base"),
    ("Distribute the annotation budget across classes.", {"allocate_budget"}, "base"),
    ("Check the accuracy, then recalibrate.", {"check_acc", "adapt"}, "base"),
    ("Detect drift and report confused classes.", {"detect_drift", "confusion_probe"}, "base"),
    ("Find unknown signals and report accuracy.", {"openset_reject", "check_acc"}, "base"),
    ("Probe confusions then allocate the budget.", {"confusion_probe", "allocate_budget"}, "base"),
    ("Recalibrate and then check accuracy.", {"adapt", "check_acc"}, "base"),
    ("Detect drift then adapt.", {"detect_drift", "adapt"}, "base"),
    # ---- extra BASE: paraphrases + multi-tool COMPOSITIONS (give the router a fair, larger
    #      training set INCLUDING composition patterns, so the win is not "too few examples") ----
    ("How accurate is the recognizer right now?", {"check_acc"}, "base"),
    ("Has the signal distribution shifted recently?", {"detect_drift"}, "base"),
    ("Tell me the most error-prone class pairs.", {"confusion_probe"}, "base"),
    ("Are we seeing signals outside the known set?", {"openset_reject"}, "base"),
    ("Update the model to current conditions.", {"adapt"}, "base"),
    ("Plan how to spend the labeling budget.", {"allocate_budget"}, "base"),
    ("Measure accuracy and detect drift.", {"check_acc", "detect_drift"}, "base"),
    ("Find unknowns and recalibrate.", {"openset_reject", "adapt"}, "base"),
    ("Report confusions and accuracy.", {"confusion_probe", "check_acc"}, "base"),
    ("Detect drift, then allocate budget.", {"detect_drift", "allocate_budget"}, "base"),
    ("Check unknowns and confusions.", {"openset_reject", "confusion_probe"}, "base"),
    ("Recalibrate and reallocate the budget.", {"adapt", "allocate_budget"}, "base"),
    ("Assess accuracy, drift, and confusions.", {"check_acc", "detect_drift", "confusion_probe"}, "base"),
    ("Reject unknowns, recalibrate, and check accuracy.", {"openset_reject", "adapt", "check_acc"}, "base"),
    # ---- NOVEL / COMPOUND (T3/T4) ----
    ("Something is degrading recognition this hour 鈥?figure out the cause and fix it.",
     {"check_acc", "detect_drift", "confusion_probe", "adapt"}, "novel"),
    ("We're seeing weird emissions and our accuracy tanked; investigate end to end.",
     {"check_acc", "openset_reject", "detect_drift", "adapt"}, "novel"),
    ("Prioritize labeling effort on whatever class pair is hurting us most at low SNR.",
     {"confusion_probe", "allocate_budget"}, "novel"),
    ("Flag new emitters and make sure we don't waste labels trying to classify them.",
     {"openset_reject", "allocate_budget"}, "novel"),
    ("If there's drift, retrain; either way tell me the confused classes afterward.",
     {"detect_drift", "adapt", "confusion_probe"}, "novel"),
    ("Before spending any budget, confirm the model is actually underperforming.",
     {"check_acc", "allocate_budget"}, "novel"),
    ("Diagnose the recognition problem but do NOT touch the model yet.",
     {"check_acc", "detect_drift", "confusion_probe"}, "novel"),
    ("Are unknown signals the reason accuracy dropped, or is it drift?",
     {"openset_reject", "detect_drift", "check_acc"}, "novel"),
    ("Clean up the unknowns, rebalance the budget, then recalibrate.",
     {"openset_reject", "allocate_budget", "adapt"}, "novel"),
    ("Audit everything that could be wrong with the deployed recognizer right now.",
     {"check_acc", "detect_drift", "confusion_probe", "openset_reject"}, "novel"),
    ("Triage: what's broken, and what's the cheapest fix?",
     {"check_acc", "detect_drift", "adapt"}, "novel"),
    ("Focus our labeling on the hardest confusions and then update the model.",
     {"confusion_probe", "allocate_budget", "adapt"}, "novel"),
    ("Don't retrain blindly 鈥?first check if it's drift or just unknown signals.",
     {"detect_drift", "openset_reject"}, "novel"),
    ("Give me a full health report of the recognizer without changing anything.",
     {"check_acc", "detect_drift", "confusion_probe", "openset_reject"}, "novel"),
    ("Spend the budget where the confusions are worst, but only if accuracy is low.",
     {"check_acc", "confusion_probe", "allocate_budget"}, "novel"),
    ("New signal types showed up; reallocate labels and refresh the model.",
     {"openset_reject", "allocate_budget", "adapt"}, "novel"),
    ("Is performance loss from drift? If so fix it and verify it worked.",
     {"detect_drift", "adapt", "check_acc"}, "novel"),
    ("Map out the failure modes and rejected signals before any retraining.",
     {"confusion_probe", "openset_reject"}, "novel"),
    # ---- expansion: more BASE paraphrases ----
    ("Give me the current accuracy figure.", {"check_acc"}, "base"),
    ("Is there concept drift right now?", {"detect_drift"}, "base"),
    ("Which classes does the model mix up?", {"confusion_probe"}, "base"),
    ("Spot any unrecognized signals.", {"openset_reject"}, "base"),
    ("Refresh the model on recent data.", {"adapt"}, "base"),
    ("Set the labeling priorities across classes.", {"allocate_budget"}, "base"),
    # ---- expansion: more NOVEL/compound (diverse compositions, conditionals, NL constraints) ----
    ("Run a complete diagnostic but hold off on any model changes.",
     {"check_acc", "detect_drift", "confusion_probe", "openset_reject"}, "novel"),
    ("Figure out whether to retrain or just reject unknowns, then do whichever is right.",
     {"detect_drift", "openset_reject", "adapt"}, "novel"),
    ("Our metrics look off 鈥?give me accuracy, drift status, and the worst confusion.",
     {"check_acc", "detect_drift", "confusion_probe"}, "novel"),
    ("Spend labels on the worst confusion, but only after ruling out drift.",
     {"detect_drift", "confusion_probe", "allocate_budget"}, "novel"),
    ("Check for new emitters and confused pairs, then rebalance the budget.",
     {"openset_reject", "confusion_probe", "allocate_budget"}, "novel"),
    ("Is the accuracy drop from unknown signals? Investigate and recalibrate if not.",
     {"openset_reject", "check_acc", "adapt"}, "novel"),
    ("Identify every issue with the recognizer and then recalibrate it.",
     {"check_acc", "detect_drift", "confusion_probe", "openset_reject", "adapt"}, "novel"),
    ("Just tell me what's wrong; don't fix anything yet.",
     {"check_acc", "detect_drift", "confusion_probe", "openset_reject"}, "novel"),
    ("Detect drift; if present, retrain and confirm accuracy improved.",
     {"detect_drift", "adapt", "check_acc"}, "novel"),
    ("Put the labeling budget where accuracy is hurting most.",
     {"check_acc", "confusion_probe", "allocate_budget"}, "novel"),
    ("Look for unknowns and drift together before deciding anything.",
     {"openset_reject", "detect_drift"}, "novel"),
    ("Recalibrate, then verify via accuracy and confusions.",
     {"adapt", "check_acc", "confusion_probe"}, "novel"),
    ("Audit the model end to end, then apply the cheapest effective fix.",
     {"check_acc", "detect_drift", "confusion_probe", "openset_reject", "adapt"}, "novel"),
    ("Reject the unknowns and move their budget to the confused classes.",
     {"openset_reject", "confusion_probe", "allocate_budget"}, "novel"),
    ("Before retraining, rule out that it's just unknowns or a confusion spike.",
     {"openset_reject", "confusion_probe", "detect_drift"}, "novel"),
    ("Report drift and unknown-signal status, then recalibrate if either is bad.",
     {"detect_drift", "openset_reject", "adapt"}, "novel"),
    ("What's the deployed model's health and what should we fix first?",
     {"check_acc", "detect_drift", "confusion_probe", "openset_reject"}, "novel"),
    ("Trace the recognition failure to drift or novelty, then act on it.",
     {"detect_drift", "openset_reject", "adapt"}, "novel"),
    ("Allocate labels to the hardest pair and recalibrate once that's done.",
     {"confusion_probe", "allocate_budget", "adapt"}, "novel"),
    ("Diagnose accuracy and confusions, allocate the budget, and recalibrate.",
     {"check_acc", "confusion_probe", "allocate_budget", "adapt"}, "novel"),
    ("Comprehensive check, then a targeted fix for the single worst problem.",
     {"check_acc", "detect_drift", "confusion_probe", "openset_reject", "adapt"}, "novel"),
    ("Confirm the model is underperforming before spending any labeling budget.",
     {"check_acc", "allocate_budget"}, "novel"),
]


def parse_tools(t):
    found = set()
    m = re.search(r"TOOLS?:\s*([a-z_,\s]+)", t, re.I)
    span = m.group(1) if m else t
    for tool in TOOLS:
        if re.search(r"\b" + tool + r"\b", span, re.I): found.add(tool)
    if not found:  # fallback: scan whole text
        for tool in TOOLS:
            if re.search(r"\b" + tool + r"\b", t, re.I): found.add(tool)
    return found


def f1(pred, gt):
    if not pred and not gt: return 1.0
    tp = len(pred & gt); p = tp / len(pred) if pred else 0.0; r = tp / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai", choices=["openai", "hf", "none"])
    ap.add_argument("--hf_model_path", default="models/gemma-4-12B-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ext_json", default="", help="JSON [[intent,[tools]],...] of EXTERNALLY-authored intents (tier 'ext')")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config, [])

    bank = list(BANK)
    if args.ext_json:
        for intent, tools in json.load(open(args.ext_json)):
            bank.append((intent, set(tools), "ext"))
    base = [b for b in bank if b[2] == "base"]; novel = [b for b in bank if b[2] == "novel"]

    # ---- learned routers, trained on BASE only ----
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    Xb = [b[0] for b in base]; vec = TfidfVectorizer().fit(Xb); Mb = vec.transform(Xb)
    def router_nn(intent):
        v = vec.transform([intent]); sims = (Mb @ v.T).toarray().ravel()
        return base[int(sims.argmax())][1]
    # char-ngram NN router (paraphrase-robust without a sentence-embedding model)
    cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(Xb); Cb = cvec.transform(Xb)
    def router_char(intent):
        v = cvec.transform([intent]); sims = (Cb @ v.T).toarray().ravel()
        return base[int(sims.argmax())][1]
    ml = {}
    for tool in TOOLS:
        y = [1 if tool in b[1] else 0 for b in base]
        ml[tool] = LogisticRegression(max_iter=500).fit(Mb, y) if sum(y) and sum(y) < len(y) else None
    def router_ml(intent):
        v = vec.transform([intent]); out = set()
        for tool in TOOLS:
            if ml[tool] is not None and ml[tool].predict(v)[0] == 1: out.add(tool)
        return out or router_nn(intent)
    # STRONGEST offline router: char-ngram features (paraphrase-robust) + per-tool logistic (composes)
    Cml = {}
    for tool in TOOLS:
        y = [1 if tool in b[1] else 0 for b in base]
        Cml[tool] = LogisticRegression(max_iter=500).fit(Cb, y) if sum(y) and sum(y) < len(y) else None
    def router_ml_char(intent):
        v = cvec.transform([intent]); out = set()
        for tool in TOOLS:
            if Cml[tool] is not None and Cml[tool].predict(v)[0] == 1: out.add(tool)
        return out or router_char(intent)
    # ZERO-SHOT tool-DESCRIPTION router: scores the intent against each tool's DESCRIPTION
    # (same descriptions the LLM sees), NO base-intent training; threshold calibrated on base.
    # Addresses the "router has an unfair info split" objection.
    TOOL_DESC = {
        "check_acc": "report the current classifier accuracy and performance level",
        "detect_drift": "detect concept drift, distribution shift or change over time",
        "confusion_probe": "report which modulation classes are most confused, errors between classes",
        "openset_reject": "detect and flag unknown, novel, out-of-distribution or unrecognized signals and emitters",
        "adapt": "update, recalibrate, retrain or refresh the classifier model to fix it",
        "allocate_budget": "allocate or spend the labeling budget across classes, prioritize labels",
    }
    dvec = TfidfVectorizer().fit([TOOL_DESC[t] for t in TOOLS] + Xb)
    Dt = dvec.transform([TOOL_DESC[t] for t in TOOLS])
    def _desc_sims(intent):
        v = dvec.transform([intent]); return (Dt @ v.T).toarray().ravel()
    # calibrate threshold on BASE for best F1
    best_tau, best_f1b = 0.05, -1
    for tau in [x / 100 for x in range(2, 40, 2)]:
        fs = []
        for b in base:
            s = _desc_sims(b[0]); pred = {TOOLS[k] for k in range(len(TOOLS)) if s[k] >= tau} or {TOOLS[int(s.argmax())]}
            fs.append(f1(pred, b[1]))
        mf = sum(fs) / len(fs)
        if mf > best_f1b: best_f1b, best_tau = mf, tau
    def router_desc(intent):
        s = _desc_sims(intent); return {TOOLS[k] for k in range(len(TOOLS)) if s[k] >= best_tau} or {TOOLS[int(s.argmax())]}
    from collections import Counter
    fixed_set = Counter(frozenset(b[1]) for b in base).most_common(1)[0][0]

    # ---- LLM agent ----
    chat = None
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(s, u):
            r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": s}, {"role": "user", "content": u}],
                temperature=0.0, max_tokens=120, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
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
            with torch.no_grad(): out = mdl.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        model = "none"

    SYS = ("You are an operations agent for an online modulation-recognition system. " + TOOL_API +
           "\nGiven an operator request, output the MINIMAL set of tools needed to fulfill it. "
           "Think briefly, then end with 'TOOLS: <comma-separated tool names>'.")

    def llm_tools(intent):
        if chat is None: return set()
        return parse_tools(chat(SYS, f"Operator request: \"{intent}\"\nWhich tools are needed? TOOLS: .."))

    methods = {"fixed": lambda i: set(fixed_set), "router_nn": router_nn, "router_char": router_char,
               "router_ml": router_ml, "router_ml_char": router_ml_char, "router_desc": router_desc}
    if chat is not None: methods["llm"] = llm_tools

    tiers = sorted({b[2] for b in bank})
    res = {m: {t: [] for t in tiers} for m in methods}
    dump = []
    for intent, gt, tier in bank:
        row = {"intent": intent, "tier": tier, "gt": sorted(gt)}
        for m, fn in methods.items():
            pred = fn(intent); res[m][tier].append(f1(pred, gt))
            row[m] = sorted(pred)
        dump.append(row)

    rep = {"model": model, "n_base": len(base), "n_novel": len(novel), "tiers": tiers,
           "metrics": {m: {t: {"f1_mean": float(np.mean(res[m][t])) if res[m][t] else None, "n": len(res[m][t])} for t in tiers} for m in methods},
           "dump": dump}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"agent_intent_{tag}.json"), "w"), indent=2)
    print(f"\n[AGENT-INTENT {model}] tool-set F1 by tier {tiers}:")
    for m in methods:
        r = rep["metrics"][m]
        print(f"  {m:14s} " + " ".join(f"{t}={r[t]['f1_mean']:.3f}" for t in tiers if r[t]['f1_mean'] is not None))
    print("[WIN if] llm novel >> router_nn/router_ml novel (zero-shot generalization to unseen NL intents),")
    print("         while llm base ~>= routers base. If routers match novel -> LLM reducible -> KILL.", flush=True)
    open(os.path.join(out_dir, f"AGENT_INTENT_DONE_{tag}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
