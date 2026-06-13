#!/usr/bin/env python
"""Open-world RF triage gate: does an LLM use REAL-WORLD band knowledge to disambiguate
low-SNR AMR where signal-only and from-data models structurally cannot?

Setup (RML2018): a CNN signal-classifier (trained on the modulations, all SNR) gives
top-k probs. TEST = low-SNR signals tagged with HELD-OUT bands (band->modulation mapping
absent from baseline training; some test modulations not even in the GBDT's train targets).
  - cnn_signal : argmax CNN probs (fails at low SNR)
  - gbdt       : GBDT([probs, band]) trained on TRAIN bands (test band unseen -> fails)
  - table_train: lookup in TRAIN band table (test bands missing -> fails)
  - table_full : lookup in FULL real table (knowledge-given UPPER BOUND)
  - llm        : freq + CNN top-3 -> recall what operates there -> name (world knowledge)
WIN: llm >> cnn_signal AND >> gbdt AND llm ~>= table_full (LLM recovers band priors zero-shot).
Also a direct test of whether the LLM HAS the niche RF band knowledge (small model may not).
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier
from amrl.rf_systems import TRAIN_BANDS, TEST_BANDS, ALL_BANDS, band_text


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--test_snr_max", type=int, default=0)   # low-SNR test (CNN unreliable)
    ap.add_argument("--per_band_test", type=int, default=16)
    ap.add_argument("--backend", default="openai", choices=["openai", "hf"])
    ap.add_argument("--hf_model_path", default="models/gemma-4-12B-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample",
                                     "dataset.test_per_class=200"])
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes
    # modulation universe = all band modulations
    mods = sorted({m for (_, m, _) in ALL_BANDS.values()})
    mods = [m for m in mods if m in names]
    midx = {m: i for i, m in enumerate(mods)}
    cls_ids = {names.index(m): midx[m] for m in mods}      # global class id -> local 0..k
    print(f"== open-world gate == mods={mods}", flush=True)

    # signal classifier CNN on these modulations, all SNR
    def subset(idxs):
        keep = idxs[np.isin(data.y[idxs], [names.index(m) for m in mods])]
        return keep
    tr = subset(data.train_idx)
    Xtr = data.X[tr]; ytr = np.array([cls_ids[int(c)] for c in data.y[tr]])
    cnn = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(mods))
    set_seed(cfg.seed); print("[train CNN signal classifier]", flush=True); cnn.fit(Xtr, ytr, args.epochs)

    te = data.train_idx  # draw test/gbdt-data from train pool (disjoint from cnn? cnn trained on train; use test_idx)
    te = data.test_idx
    # GBDT training data: TRAIN-band modulations, tagged with their train band, across SNR
    gbdt = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xg, yg = [], []
        train_band_mods = {m for (_, m, _) in TRAIN_BANDS.values()}
        band_of = {}  # global class id -> assigned train band freq (first match)
        for f, (sys_, m, _) in TRAIN_BANDS.items():
            band_of.setdefault(names.index(m), f)
        samp = te[np.isin(data.y[te], [names.index(m) for m in train_band_mods])]
        samp = rng.permutation(samp)[:2000]
        pg = cnn.predict_proba(data.X[samp])
        for k, gi in enumerate(samp):
            f = band_of[int(data.y[gi])]
            Xg.append(np.concatenate([pg[k], [np.log10(f)]])); yg.append(names[int(data.y[gi])])
        gbdt = HistGradientBoostingClassifier(max_iter=150, max_depth=4)
        gbdt.fit(np.array(Xg), np.array(yg))
    except Exception as e:
        print(f"[warn] GBDT baseline skipped ({type(e).__name__}: {e})", flush=True)

    # TEST set: held-out bands, LOW SNR
    test = []
    for f, (sys_, m, note) in TEST_BANDS.items():
        cand = te[(data.y[te] == names.index(m)) & (data.snr[te] <= args.test_snr_max)]
        for gi in rng.permutation(cand)[:args.per_band_test]:
            test.append((int(gi), f, m))
    Xte = data.X[[t[0] for t in test]]
    pte = cnn.predict_proba(Xte)
    print(f"[test] {len(test)} samples over {len(TEST_BANDS)} held-out bands, SNR<={args.test_snr_max}", flush=True)

    # backend
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(s, u):
            r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": s}, {"role": "user", "content": u}],
                temperature=0.0, max_tokens=250, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.hf_model_path)
        mdl = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16, device_map=args.device).eval()
        model = os.path.basename(args.hf_model_path)
        def chat(s, u):
            enc = tok.apply_chat_template([{"role": "user", "content": s + "\n\n" + u}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=250, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    SYS = ("You are an RF spectrum analyst. A modulation classifier is UNRELIABLE at this low SNR. Use your "
           "knowledge of which real-world radio systems operate at the given carrier frequency, and what "
           "modulation those systems use, to identify the most likely modulation. End with 'ANSWER: <one candidate>'.")

    def llm_pred(probs, f):
        top = np.argsort(-probs)[:3]
        topstr = ", ".join(f"{mods[t]}({probs[t]:.2f})" for t in top)
        u = (f"Observed: {band_text(f)}. Unreliable classifier top-3: {topstr}.\n"
             f"Candidate modulations: {', '.join(mods)}.\n"
             "What modulation is most likely, given what operates at this frequency? ANSWER: <one candidate>.")
        txt = chat(SYS, u)
        mt = re.search(r"ANSWER:\s*([A-Za-z0-9\-]+)", txt)
        cand = mt.group(1) if mt else txt
        for m in mods:
            if m.lower() == cand.lower(): return m
        for m in mods:
            if m.lower() in txt.lower(): return m
        return None

    n = len(test)
    cnn_c = gbdt_c = tt_c = tf_c = llm_c = unp = 0
    for k, (gi, f, m) in enumerate(test):
        if mods[int(pte[k].argmax())] == m: cnn_c += 1
        if gbdt is not None and gbdt.predict([np.concatenate([pte[k], [np.log10(f)]])])[0] == m: gbdt_c += 1
        if TRAIN_BANDS.get(f, (None, None, None))[1] == m: tt_c += 1
        if ALL_BANDS.get(f, (None, None, None))[1] == m: tf_c += 1
        pr = llm_pred(pte[k], f)
        if pr is None: unp += 1
        elif pr == m: llm_c += 1
        if (k + 1) % 20 == 0: print(f"  ...{k+1}/{n}", flush=True)

    rep = {"dataset": args.dataset, "model": model, "n_test": n, "test_snr_max": args.test_snr_max,
           "chance": 1.0 / len(mods), "test_bands": {str(f): TEST_BANDS[f][1] for f in TEST_BANDS},
           "cnn_signal_acc": cnn_c / n, "gbdt_acc": gbdt_c / n, "table_train_acc": tt_c / n,
           "table_full_acc(upper)": tf_c / n, "llm_acc": llm_c / n, "n_unparsed": unp}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"openworld_gate_{tag}{args.dataset}.json"), "w"), indent=2)
    print(f"\n[OPENWORLD {model}] llm={rep['llm_acc']:.3f} | cnn_signal={rep['cnn_signal_acc']:.3f} "
          f"gbdt={rep['gbdt_acc']:.3f} table_train={rep['table_train_acc']:.3f} "
          f"table_full(upper)={rep['table_full_acc(upper)']:.3f} | chance={rep['chance']:.3f} unparsed={unp}", flush=True)
    print("[WIN if] llm >> cnn_signal AND >> gbdt AND llm ~>= table_full", flush=True)
    open(os.path.join(out_dir, f"OPENWORLD_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
