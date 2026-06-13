#!/usr/bin/env python
"""RAG path for open-world AMR: can a SMALL model (Qwen3-8B) + RAG over a range-based
spectrum-allocation KB recover the frontier model's world knowledge -- and does it
beat the DETERMINISTIC baseline that uses the SAME KB?

Reads the dumped fusion cases (with full CNN probs), then computes:
  - range_det   : retrieve allocation range -> predict its PRIMARY mod (KB, no signal)
  - range_sig   : retrieve range -> KB-restricted signal argmax (KB + signal, NO LLM)  <-- the fair baseline
  - rag_freqonly: Qwen3-8B + retrieved KB text, frequency only
  - rag_fusion  : Qwen3-8B + retrieved KB text + CNN top-3 (signal fusion)
Verdict: (a) does RAG let the small model reach frontier-level (vs its 0.34 baked-in)?
         (b) does ANY LLM beat range_sig? If not, the value is KB+signal, not the LLM.
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from amrl.config import load_config
from amrl.spectrum_kb import retrieve_text, range_det_pred, range_sig_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(ROOT, "..", "results", "openworld_fusion_cases_f1_rml2018.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "results", "openworld_rag_scored_f1.json"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    d = json.load(open(args.cases)); cases = d["cases"]; summ = d["summary"]; mods = summ["mods"]

    # --- deterministic range-KB baselines (no LLM) ---
    rd_c = rs_c = 0
    for c in cases:
        if range_det_pred(c["freq"]) == c["true"]: rd_c += 1
        if range_sig_pred(c["freq"], c["cnn_probs"], mods) == c["true"]: rs_c += 1
    n = len(cases)

    # --- RAG-Qwen3-8B via vLLM ---
    from openai import OpenAI
    cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
    def chat(sysp, usr):
        r = cli.chat.completions.create(model=model,
            messages=[{"role": "system", "content": sysp}, {"role": "user", "content": usr}],
            temperature=0.0, max_tokens=200, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return r.choices[0].message.content or ""

    SYS = ("You are an RF spectrum analyst. Use the RETRIEVED spectrum-allocation knowledge "
           "(which real-world service and modulations occupy the carrier frequency's band) to "
           "identify the most likely modulation of a low-SNR signal. End with 'ANSWER: <one candidate>'.")

    def ask(c, fusion):
        kb = retrieve_text(c["freq"])
        usr = f"Carrier frequency: {c['freq']:g} MHz.\n{kb}\n"
        if fusion:
            usr += "Unreliable low-SNR classifier top-3: " + ", ".join(f"{m}({p})" for m, p in c["cnn_top3"]) + ".\n"
        usr += (f"Candidate modulations: {', '.join(c['candidates'])}.\n"
                "Which modulation is most likely? ANSWER: <one candidate>.")
        txt = chat(SYS, usr)
        m = re.search(r"ANSWER:\s*\**([A-Za-z0-9\-]+)", txt)
        cand = m.group(1) if m else txt
        for x in c["candidates"]:
            if x.lower() == cand.lower(): return x
        for x in c["candidates"]:
            if x.lower() in txt.lower(): return x
        return None

    fo_c = fu_c = fo_unp = fu_unp = 0
    fo_bt = {"unique": [0, 0], "ambiguous": [0, 0]}; fu_bt = {"unique": [0, 0], "ambiguous": [0, 0]}
    for k, c in enumerate(cases):
        bt = c["band_type"]
        pfo = ask(c, False); pfu = ask(c, True)
        fo_bt[bt][1] += 1; fu_bt[bt][1] += 1
        if pfo is None: fo_unp += 1
        elif pfo == c["true"]: fo_c += 1; fo_bt[bt][0] += 1
        if pfu is None: fu_unp += 1
        elif pfu == c["true"]: fu_c += 1; fu_bt[bt][0] += 1
        if (k + 1) % 28 == 0: print(f"  ...{k+1}/{n}", flush=True)

    rep = dict(summ)
    rep.update({"range_det_acc": rd_c / n, "range_sig_acc": rs_c / n,
                "rag_freqonly_acc": fo_c / n, "rag_freqonly_unparsed": fo_unp,
                "rag_freqonly_unique": fo_bt["unique"][0] / max(1, fo_bt["unique"][1]),
                "rag_freqonly_ambiguous": fo_bt["ambiguous"][0] / max(1, fo_bt["ambiguous"][1]),
                "rag_fusion_acc": fu_c / n, "rag_fusion_unparsed": fu_unp,
                "rag_fusion_unique": fu_bt["unique"][0] / max(1, fu_bt["unique"][1]),
                "rag_fusion_ambiguous": fu_bt["ambiguous"][0] / max(1, fu_bt["ambiguous"][1]),
                "rag_model": model})
    out_dir = os.path.dirname(args.out); os.makedirs(out_dir, exist_ok=True)
    json.dump({"report": rep}, open(args.out, "w"), indent=2)
    print("\n=== OPEN-WORLD RAG (small model + range-KB) ===")
    print(f"n={n} (unique={summ['n_unique']} ambig={summ['n_ambiguous']}) chance={summ['chance']:.3f}")
    print(f"  fixed_table(point) = {summ['fixed_table_acc']:.3f}")
    print(f"  signal_only        = {summ['signal_only_acc']:.3f}")
    print(f"  gbdt               = {summ['gbdt_acc']:.3f}")
    print(f"  range_det (KB,nosig)= {rd_c/n:.3f}")
    print(f"  range_sig (KB+sig)  = {rs_c/n:.3f}   <-- FAIR baseline the LLM must beat")
    print(f"  rag_freqonly (Qwen) = {fo_c/n:.3f}  [uniq {fo_bt['unique'][0]/max(1,fo_bt['unique'][1]):.3f} | amb {fo_bt['ambiguous'][0]/max(1,fo_bt['ambiguous'][1]):.3f}] unp={fo_unp}")
    print(f"  rag_fusion   (Qwen) = {fu_c/n:.3f}  [uniq {fu_bt['unique'][0]/max(1,fu_bt['unique'][1]):.3f} | amb {fu_bt['ambiguous'][0]/max(1,fu_bt['ambiguous'][1]):.3f}] unp={fu_unp}")
    open(os.path.join(out_dir, "RAG_DONE_f1.flag"), "w").write("done")


if __name__ == "__main__":
    main()
