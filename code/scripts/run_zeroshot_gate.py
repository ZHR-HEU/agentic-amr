#!/usr/bin/env python
"""Zero-shot headroom gate for direction C (LLM knowledge for unseen modulation).

Core question: given interpretable RF descriptors, can a FROZEN LLM identify the
modulation scheme ZERO-SHOT (no examples), beating chance and meaningfully covering
classes a closed-set model couldn't? If the LLM's textual modulation knowledge does
not ground to these descriptors, the open-set direction is dead -> stop early.

Compares: chance (1/#classes) ; trained 1-NN on descriptors (reference upper-ish bound) ;
LLM zero-shot. Also reports accuracy on a held-out 'unseen' class set.
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
from amrl.rf_features import compute_descriptors, render_descriptor_text, FEATURE_LEGEND, FEATURE_NAMES


def _lvl(pct):
    return "very-low" if pct < 15 else "low" if pct < 40 else "medium" if pct < 60 else "high" if pct < 85 else "very-high"


def render_paramcard(vec, Dref):
    """Semantic ESTIMATED-PARAMETER card (envelope / freq-mod / phase-order / family hint)
    鈥?physical parameters the LLM has strong knowledge about, vs abstract cumulant numbers."""
    pct = {nm: float((Dref[:, j] < vec[j]).mean() * 100) for j, nm in enumerate(FEATURE_NAMES)}
    amp, ifr, phr = pct["amp_std"], pct["ifreq_std"], pct["phase_resid_std"]
    envelope = ("constant (no amplitude information)" if amp < 35
                else "strongly amplitude-varying" if amp > 70 else "moderately amplitude-varying")
    freqmod = ("strong (frequency-modulated)" if ifr > 70 else "mild" if ifr > 45 else "negligible")
    if ifr > 68:
        fam = "frequency-modulated family (e.g. 2FSK/4FSK/GMSK/FM)"
    elif amp > 68:
        fam = "amplitude/QAM family (e.g. 16/32/64/128/256-QAM, PAM, ASK)"
    elif amp < 38 and phr > 45:
        fam = "phase-modulated family (e.g. BPSK/QPSK/8PSK/16PSK and APSK)"
    else:
        fam = "uncertain (possibly analog AM/SSB/DSB or a low-order/near-constant scheme)"
    return ("Estimated signal parameters (power-normalized; percentiles vs population):\n"
            f"- envelope: {envelope} (amplitude-variation P{amp:.0f})\n"
            f"- frequency modulation: {freqmod} (instantaneous-freq spread P{ifr:.0f})\n"
            f"- phase-state spread: P{phr:.0f} (higher => more discrete phase states => higher PSK order)\n"
            f"- higher-order cumulants: |C40| P{pct['|C40|']:.0f}, |C42| P{pct['|C42|']:.0f} "
            f"(near-zero C40 suggests QAM; large C40 suggests low-order PSK)\n"
            f"- spectral peak/flatness: P{pct['spec_peak2mean']:.0f}/P{pct['spec_flatness']:.0f} "
            f"(tonal vs broadband; analog vs digital cues)\n"
            f"- ESTIMATED FAMILY HINT: {fam}\n")


def render_rich(vec, Dref):
    """Interpretable rendering: per-feature value + population percentile + qualitative tags,
    so the LLM can apply textbook modulation knowledge (it has no absolute scale otherwise)."""
    pct = {nm: float((Dref[:, j] < vec[j]).mean() * 100) for j, nm in enumerate(FEATURE_NAMES)}
    q = []
    q.append(f"envelope={'constant' if pct['amp_std'] < 40 else 'amplitude-varying'} "
             f"(amp_std {_lvl(pct['amp_std'])})")
    q.append(f"freq_variation={_lvl(pct['ifreq_std'])} (FSK/FM if high)")
    q.append(f"phase_spread={_lvl(pct['phase_resid_std'])} (more PSK states if higher)")
    q.append(f"|C40|={_lvl(pct['|C40|'])}  |C42|={_lvl(pct['|C42|'])}  spec_peak2mean={_lvl(pct['spec_peak2mean'])}")
    lines = "  ".join(f"{nm}={vec[j]:.3g}(P{pct[nm]:.0f})" for j, nm in enumerate(FEATURE_NAMES))
    return "Qualitative: " + "; ".join(q) + "\nDescriptors (value, population-percentile):\n" + lines

UNSEEN_DEFAULT = "QAM64,WBFM,GFSK,PAM4"

SYS = ("You are an expert RF/communications signal analyst. Given interpretable, "
       "power-normalized descriptors of a received signal, identify its modulation "
       "scheme using your knowledge of how modulation families manifest in these "
       "descriptors. Reason briefly if needed but END with a line 'ANSWER: <name>' "
       "where <name> is EXACTLY one candidate from the list.")


def llm_identify(client, model, cfg, desc_text, names):
    user = (FEATURE_LEGEND + "\nSignal descriptors:\n" + desc_text +
            "\n\nCandidate modulations: " + ", ".join(names) +
            "\nIdentify the modulation. End with 'ANSWER: <one candidate>'.")
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": user}],
            temperature=0.0, max_tokens=300,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        txt = r.choices[0].message.content or ""
    except Exception as e:
        return None, f"ERR:{type(e).__name__}"
    m = re.search(r"ANSWER:\s*([A-Za-z0-9\-]+)", txt)
    cand = m.group(1) if m else txt
    # match against names (case-insensitive, allow substring)
    low = cand.lower()
    for nm in names:
        if nm.lower() == low:
            return nm, txt[:80]
    for nm in names:
        if nm.lower() in txt.lower():
            return nm, txt[:80]
    return None, txt[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2016")
    ap.add_argument("--snr_min", type=int, default=12)
    ap.add_argument("--per_class", type=int, default=8)
    ap.add_argument("--ref_per_class", type=int, default=40)
    ap.add_argument("--param_card", action="store_true", help="use semantic estimated-parameter card")
    ap.add_argument("--unseen", default=UNSEEN_DEFAULT)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=none"])
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg)
    names = data.classes
    unseen = set(x.strip() for x in args.unseen.split(",") if x.strip())
    print(f"== zero-shot gate == dataset={args.dataset} classes={len(names)} "
          f"snr>={args.snr_min} per_class={args.per_class} unseen={sorted(unseen)}", flush=True)

    hi = np.where(data.snr >= args.snr_min)[0]
    by_class = {c: hi[data.y[hi] == c] for c in range(len(names))}

    # test + reference (disjoint) signals per class
    test_idx, ref_idx = [], []
    for c, idxs in by_class.items():
        idxs = rng.permutation(idxs)
        test_idx += list(idxs[:args.per_class])
        ref_idx += list(idxs[args.per_class:args.per_class + args.ref_per_class])
    test_idx = np.array(test_idx); ref_idx = np.array(ref_idx)

    Dtest = compute_descriptors(data.X[test_idx]); ytest = data.y[test_idx]
    Dref = compute_descriptors(data.X[ref_idx]); yref = data.y[ref_idx]

    # standardize by reference stats (for 1-NN); keep raw for LLM text
    mu, sd = Dref.mean(0), Dref.std(0) + 1e-9
    Zt = (Dtest - mu) / sd; Zr = (Dref - mu) / sd

    # 1-NN reference (TRAINED on ref labels 鈥?not zero-shot)
    nn_pred = np.array([yref[np.argmin(((Zr - z) ** 2).sum(1))] for z in Zt])
    nn_acc = float((nn_pred == ytest).mean())

    # LLM zero-shot
    from openai import OpenAI
    client = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY")
    llm_pred, n_unparsed = [], 0
    for i in range(len(test_idx)):
        desc_text = render_paramcard(Dtest[i], Dref) if args.param_card else render_rich(Dtest[i], Dref)
        pred, _ = llm_identify(client, cfg.controller.model, cfg, desc_text, names)
        if pred is None:
            n_unparsed += 1
            llm_pred.append(-1)
        else:
            llm_pred.append(names.index(pred))
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(test_idx)} LLM calls", flush=True)
    llm_pred = np.array(llm_pred)
    llm_acc = float((llm_pred == ytest).mean())

    # per-class + seen/unseen breakdown
    unseen_ids = {names.index(u) for u in unseen if u in names}
    seen_mask = np.array([y not in unseen_ids for y in ytest])
    llm_seen = float((llm_pred[seen_mask] == ytest[seen_mask]).mean()) if seen_mask.any() else float("nan")
    llm_unseen = float((llm_pred[~seen_mask] == ytest[~seen_mask]).mean()) if (~seen_mask).any() else float("nan")
    per_class = {}
    for c in range(len(names)):
        m = ytest == c
        per_class[names[c]] = float((llm_pred[m] == c).mean()) if m.any() else float("nan")

    chance = 1.0 / len(names)
    gate = bool(llm_acc >= max(0.25, 2.5 * chance) and (np.isfinite(llm_unseen) and llm_unseen > chance))
    report = {
        "dataset": args.dataset, "snr_min": args.snr_min, "per_class": args.per_class,
        "n_test": int(len(test_idx)), "chance": chance,
        "llm_zeroshot_acc": llm_acc, "llm_seen_acc": llm_seen, "llm_unseen_acc": llm_unseen,
        "nn_reference_acc": nn_acc, "n_unparsed": n_unparsed,
        "unseen_classes": sorted(unseen), "per_class_acc": per_class, "gate_pass": gate,
    }
    out_dir = os.path.join(ROOT, cfg.eval.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"zeroshot_gate_{args.dataset}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ZS GATE {args.dataset}] chance={chance:.3f} LLM_zeroshot={llm_acc:.3f} "
          f"(seen={llm_seen:.3f} unseen={llm_unseen:.3f}) 1NN_trained_ref={nn_acc:.3f} "
          f"unparsed={n_unparsed} -> {'PASS' if gate else 'fail'}", flush=True)
    print("[per-class zero-shot acc] " + ", ".join(f"{k}:{v:.2f}" for k, v in per_class.items()), flush=True)
    open(os.path.join(out_dir, f"ZS_DONE_{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
