#!/usr/bin/env python
"""Open-world spectrum-knowledge-guided AMR 鈥?the DECISIVE minimal gate.

Tests TWO claims at once, on TABLE-UNLISTED bands + LOW SNR (a regime with no
training labels AND absent from any fixed band-plan table):

  Claim 1 (irreducible world knowledge): an LLM that REASONS over real-world
    spectrum allocation beats every from-data / lookup baseline, ALL of which
    structurally fail here:
      - signal_only : argmax CNN(IQ)            -> low SNR => unreliable
      - gbdt        : GBDT([CNN probs, freq])   -> test bands unseen => fails
      - fixed_table : lookup in TRAIN band table-> test bands UNLISTED => UNKNOWN (0)
      - table_full  : full real band table       -> knowledge UPPER BOUND, BUT it has
                       no signal, so on AMBIGUOUS bands it can only guess the set
  Claim 2 (genuine AMR, not DB-lookup): llm_fusion (freq + CNN top-k) > llm_freqonly
    (freq only) -> the IQ/signal measurably disambiguates, so it is real
    signal-aware modulation recognition, not a spectrum-database query. On the
    AMBIGUOUS band, llm_fusion can even EXCEED table_full (signal breaks the tie).

This script does NOT call the LLM. It trains the CNN, computes all non-LLM
baselines, and DUMPS test_cases_fusion.json for batched frontier-LLM scoring
(via Codex-CLI). Modulation names match RML2018.01A classes.
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier

# --- bands an LLM must REASON about (NONE are in the fixed train table below) ---
# unique: frequency alone (via world knowledge) determines the modulation
UNIQUE_BANDS = {
    157.0:  ("marine VHF maritime mobile voice band", "FM"),
    121.5:  ("aviation VHF emergency 'Guard' voice channel", "AM-DSB-WC"),
    1227.6: ("GNSS GPS L2 navigation band", "BPSK"),
    162.0:  ("maritime AIS transponder band (161.975/162.025 MHz)", "GMSK"),
}
# ambiguous: frequency narrows to a CANDIDATE SET; the signal must disambiguate
AMBIG_BANDS = {
    11700.0: ("Ku-band DVB-S2 satellite TV downlink", ["QPSK", "8PSK", "16APSK", "32APSK"]),
}
# fixed band-plan lookup table available to baselines (TRAIN bands only -> none of the
# test bands above appear here, so fixed_table returns UNKNOWN for every test sample)
FIXED_TABLE = {100.0: "FM", 940.0: "GMSK", 1575.42: "BPSK", 5800.0: "64QAM", 12000.0: "QPSK"}
# full real table = FIXED_TABLE + the true mapping of the test bands (knowledge UPPER BOUND)
FULL_TABLE = dict(FIXED_TABLE)
for f, (_, m) in UNIQUE_BANDS.items():
    FULL_TABLE[f] = m  # unique -> known exactly


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--test_snr_max", type=int, default=0)
    ap.add_argument("--per_cell", type=int, default=14)  # samples per (band) or per (ambig mod)
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample",
                                     "dataset.test_per_class=200"])
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes

    # modulation universe = all mods appearing in the test bands
    test_mods = sorted({m for (_, m) in UNIQUE_BANDS.values()} |
                       {m for (_, cs) in AMBIG_BANDS.values() for m in cs})
    test_mods = [m for m in test_mods if m in names]
    midx = {m: i for i, m in enumerate(test_mods)}
    cls_ids = {names.index(m): midx[m] for m in test_mods}
    print(f"== open-world FUSION gate == mods={test_mods}", flush=True)

    # CNN signal classifier on these mods, all SNR
    tr = data.train_idx[np.isin(data.y[data.train_idx], [names.index(m) for m in test_mods])]
    Xtr = data.X[tr]; ytr = np.array([cls_ids[int(c)] for c in data.y[tr]])
    cnn = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(test_mods))
    set_seed(cfg.seed); print("[train CNN signal classifier]", flush=True); cnn.fit(Xtr, ytr, args.epochs)
    te = data.test_idx

    # GBDT baseline: trained on FIXED_TABLE (train) bands, features [CNN probs, log10 freq]
    gbdt = None
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        band_of = {}
        for f, m in FIXED_TABLE.items():
            if m in names: band_of.setdefault(names.index(m), f)
        samp = te[np.isin(data.y[te], list(band_of.keys()))]
        samp = rng.permutation(samp)[:2500]
        if len(samp):
            pg = cnn_proba_safe(cnn, data.X[samp], test_mods)
            Xg = [np.concatenate([pg[k], [np.log10(band_of[int(data.y[gi])])]]) for k, gi in enumerate(samp)]
            yg = [names[int(data.y[gi])] for gi in samp]
            gbdt = HistGradientBoostingClassifier(max_iter=200, max_depth=5)
            gbdt.fit(np.array(Xg), np.array(yg))
    except Exception as e:
        print(f"[warn] GBDT skipped ({type(e).__name__}: {e})", flush=True)

    # ---- build test cases (low SNR, table-unlisted bands) ----
    cases = []
    for f, (ctx, m) in UNIQUE_BANDS.items():
        if m not in test_mods: continue
        cand = te[(data.y[te] == names.index(m)) & (data.snr[te] <= args.test_snr_max)]
        for gi in rng.permutation(cand)[:args.per_cell]:
            cases.append({"gi": int(gi), "freq": f, "context": ctx, "true": m,
                          "candidates": test_mods, "band_type": "unique"})
    for f, (ctx, cs) in AMBIG_BANDS.items():
        cs = [m for m in cs if m in test_mods]
        for m in cs:
            cand = te[(data.y[te] == names.index(m)) & (data.snr[te] <= args.test_snr_max)]
            for gi in rng.permutation(cand)[:args.per_cell]:
                cases.append({"gi": int(gi), "freq": f, "context": ctx, "true": m,
                              "candidates": cs, "band_type": "ambiguous"})

    Xc = data.X[[c["gi"] for c in cases]]
    pc = cnn_proba_safe(cnn, Xc, test_mods)

    # ---- non-LLM baselines + dump per-case CNN top-3 for the LLM ----
    sig_c = gbdt_c = fix_c = full_c = 0
    for k, c in enumerate(cases):
        true = c["true"]; f = c["freq"]
        top = np.argsort(-pc[k])[:3]
        c["cnn_top3"] = [[test_mods[t], round(float(pc[k][t]), 3)] for t in top]
        c["cnn_probs"] = [round(float(x), 4) for x in pc[k]]  # full vector, mods order
        c["signal_only_pred"] = test_mods[int(pc[k].argmax())]
        sig_c += (c["signal_only_pred"] == true)
        c["gbdt_pred"] = (str(gbdt.predict([np.concatenate([pc[k], [np.log10(f)]])])[0]) if gbdt is not None else None)
        gbdt_c += (c["gbdt_pred"] == true)
        c["fixed_table_pred"] = FIXED_TABLE.get(f, "UNKNOWN")  # test bands unlisted -> UNKNOWN
        fix_c += (c["fixed_table_pred"] == true)
        # table_full: unique bands -> exact; ambiguous -> table has no signal, guesses 1st candidate
        if f in FULL_TABLE:
            tf = FULL_TABLE[f]
        else:
            tf = c["candidates"][0]  # ambiguous: knowledge table knows the SET, must guess one
        c["table_full_pred"] = tf
        full_c += (tf == true)

    n = len(cases)
    summary = {"dataset": args.dataset, "n": n, "mods": test_mods, "test_snr_max": args.test_snr_max,
               "chance": 1.0 / len(test_mods),
               "n_unique": sum(c["band_type"] == "unique" for c in cases),
               "n_ambiguous": sum(c["band_type"] == "ambiguous" for c in cases),
               "signal_only_acc": sig_c / n, "gbdt_acc": gbdt_c / n,
               "fixed_table_acc": fix_c / n, "table_full_acc": full_c / n,
               "_note": "llm_freqonly / llm_fusion accuracies are filled by the Codex-CLI batch scorer"}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump({"summary": summary, "cases": cases},
              open(os.path.join(out_dir, f"openworld_fusion_cases_{tag}{args.dataset}.json"), "w"), indent=2)
    print(f"\n[FUSION baselines] n={n} (unique={summary['n_unique']} ambig={summary['n_ambiguous']}) "
          f"signal_only={sig_c/n:.3f} gbdt={gbdt_c/n:.3f} fixed_table={fix_c/n:.3f} "
          f"table_full={full_c/n:.3f} chance={1.0/len(test_mods):.3f}", flush=True)
    print("[next] score llm_freqonly & llm_fusion via Codex-CLI batch on the dumped cases", flush=True)
    open(os.path.join(out_dir, f"FUSION_CASES_DONE_{tag}{args.dataset}.flag"), "w").write("done")


def cnn_proba_safe(cnn, X, mods):
    p = cnn.predict_proba(X)
    return p


if __name__ == "__main__":
    main()
