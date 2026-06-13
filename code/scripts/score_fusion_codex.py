#!/usr/bin/env python
"""Score llm_freqonly & llm_fusion on the open-world fusion cases using a FRONTIER
model via the Codex CLI. Two-stage (codex is an npm shell script, not callable from
Windows-python subprocess, so bash drives codex in between):

  python score_fusion_codex.py --mode build     # writes per-batch prompt files
  (bash loop) codex exec ... - < prompt > resp   # frontier model answers
  python score_fusion_codex.py --mode score      # parses resp files, scores

Conditions: freqonly (freq+context only) vs fusion (freq+context+CNN top-3).
WIN: fusion >> from-data/fixed_table (world knowledge irreducible) AND
     fusion > freqonly (signal disambiguates => genuine AMR, esp. ambiguous band).
"""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "..", "results")
IO = os.path.join(RES, "fusion_io")


def build_prompt(batch, fusion):
    head = ("You are an RF spectrum analyst. For each numbered case, identify the most likely "
            "modulation of a low-SNR signal by reasoning about which real-world radio system "
            "operates at the given carrier frequency" +
            (", and which candidate is most consistent with the (unreliable) classifier hint" if fusion else "") +
            ". OUTPUT EXACTLY one line per case as '<k>: <MOD>', where <MOD> is copied verbatim "
            "from that case's candidate list. Output ONLY those lines, no commentary.\n\n")
    lines = []
    for k, c in batch:
        seg = f"Case {k} | carrier {c['freq']:g} MHz | band: {c['context']}"
        if fusion:
            seg += " | classifier top-3: " + ", ".join(f"{m}({p})" for m, p in c["cnn_top3"])
        seg += " | candidates: " + ", ".join(c["candidates"])
        lines.append(seg)
    return head + "\n".join(lines) + f"\n\nNow output the {len(batch)} lines."


def parse_resp(txt, batch):
    ans = {}
    for ln in txt.splitlines():
        m = re.match(r"\s*(?:Case\s*)?(\d+)\s*[:\-]\s*\**([A-Za-z0-9\-]+)", ln)
        if m:
            ans[int(m.group(1))] = m.group(2)
    out = {}
    for k, c in batch:
        cand = ans.get(k); pred = None
        if cand:
            for m in c["candidates"]:
                if m.lower() == cand.lower(): pred = m; break
            if pred is None:
                for m in c["candidates"]:
                    if m.lower() in cand.lower() or cand.lower() in m.lower(): pred = m; break
        out[k] = pred
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "score"], required=True)
    ap.add_argument("--cases", default=os.path.join(RES, "openworld_fusion_cases_f1_rml2018.json"))
    ap.add_argument("--out", default=os.path.join(RES, "openworld_fusion_scored_f1.json"))
    ap.add_argument("--batch", type=int, default=28)
    args = ap.parse_args()

    d = json.load(open(args.cases)); cases = d["cases"]; summ = d["summary"]
    idx_cases = list(enumerate(cases))
    batches = [idx_cases[i:i + args.batch] for i in range(0, len(idx_cases), args.batch)]
    os.makedirs(IO, exist_ok=True)

    if args.mode == "build":
        manifest = []
        for fusion in (False, True):
            cond = "fusion" if fusion else "freqonly"
            for bi, batch in enumerate(batches):
                fn = os.path.join(IO, f"{cond}_{bi}.prompt.txt")
                open(fn, "w", encoding="utf-8").write(build_prompt(batch, fusion))
                manifest.append(f"{cond}_{bi}")
        print(" ".join(manifest))  # bash reads this list
        return

    # score
    preds = {"freqonly": {}, "fusion": {}}
    for cond in ("freqonly", "fusion"):
        for bi, batch in enumerate(batches):
            fn = os.path.join(IO, f"{cond}_{bi}.resp.txt")
            txt = open(fn, encoding="utf-8", errors="replace").read() if os.path.exists(fn) else ""
            preds[cond].update(parse_resp(txt, batch))

    def score(cond):
        tot = {"all": [0, 0], "unique": [0, 0], "ambiguous": [0, 0]}; unp = 0
        for k, c in idx_cases:
            pred = preds[cond].get(k); bt = c["band_type"]
            for key in ("all", bt):
                tot[key][1] += 1
                if pred == c["true"]: tot[key][0] += 1
            if pred is None: unp += 1
        return {k: (v[0] / v[1] if v[1] else 0.0) for k, v in tot.items()}, unp

    fo, fo_unp = score("freqonly"); fu, fu_unp = score("fusion")
    rep = dict(summ)
    rep.update({"llm_freqonly_acc": fo["all"], "llm_freqonly_unique": fo["unique"], "llm_freqonly_ambiguous": fo["ambiguous"], "llm_freqonly_unparsed": fo_unp,
                "llm_fusion_acc": fu["all"], "llm_fusion_unique": fu["unique"], "llm_fusion_ambiguous": fu["ambiguous"], "llm_fusion_unparsed": fu_unp})
    json.dump({"report": rep, "preds": {k: {str(i): v for i, v in p.items()} for k, p in preds.items()}},
              open(args.out, "w"), indent=2)
    print("\n=== OPEN-WORLD FUSION (frontier via Codex) ===")
    print(f"n={summ['n']} (unique={summ['n_unique']} ambig={summ['n_ambiguous']}) chance={summ['chance']:.3f}")
    print(f"  fixed_table   = {summ['fixed_table_acc']:.3f}   (unlisted -> UNKNOWN)")
    print(f"  signal_only   = {summ['signal_only_acc']:.3f}")
    print(f"  gbdt          = {summ['gbdt_acc']:.3f}")
    print(f"  table_full    = {summ['table_full_acc']:.3f}   (point-table knowledge UB)")
    print(f"  llm_freqonly  = {fo['all']:.3f}   [unique {fo['unique']:.3f} | ambig {fo['ambiguous']:.3f}]  unparsed={fo_unp}")
    print(f"  llm_fusion    = {fu['all']:.3f}   [unique {fu['unique']:.3f} | ambig {fu['ambiguous']:.3f}]  unparsed={fu_unp}")


if __name__ == "__main__":
    main()
