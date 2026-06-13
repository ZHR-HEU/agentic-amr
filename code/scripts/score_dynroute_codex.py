#!/usr/bin/env python
"""Frontier (GPT-5.5 via Codex) replay of the dynamic-routing decisions WITH the same
in-context labeled exemplars (ICL) the GBDT trained on. Two-stage (bash drives codex):
  python score_dynroute_codex.py --mode build   # write batch prompts (subsampled novel routes)
  (bash) codex exec ... - < prompt > resp
  python score_dynroute_codex.py --mode score   # parse, score vs baselines

Answers "would a stronger model help?": does GPT-5.5 + ICL beat the fair gbdt_hist on
the zero-shot novel routes, or still cap at the information ceiling of the observable state?
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "..", "results")
IO = os.path.join(RES, "dynroute_io")
NOVEL = ["into_tunnel", "congested_canyon", "stop_go"]


def select(records, per_route):
    out = []
    for rt in NOVEL:
        rs = [r for r in records if r["route"] == rt]
        if not rs: continue
        step = max(1, len(rs) // per_route)
        out += rs[::step][:per_route]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "score"], required=True)
    ap.add_argument("--dump", default=os.path.join(RES, "dynroute_icl_dump.json"))
    ap.add_argument("--per_route", type=int, default=24)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(RES, "dynroute_frontier_scored.json"))
    args = ap.parse_args()

    d = json.load(open(args.dump, encoding="utf-8")); sel = select(d["records"], args.per_route)
    os.makedirs(IO, exist_ok=True)
    batches = [sel[i:i + args.batch] for i in range(0, len(sel), args.batch)]

    if args.mode == "build":
        man = []
        for bi, batch in enumerate(batches):
            head = ("You are routing an online modulation classifier among 4 specialists (#0 clean/MID-SNR, "
                    "#1 fading/MID-SNR, #2 clean/LOW-SNR, #3 interference/MID-SNR). Answer the independent CASES "
                    "below; each contains its own context and labeled examples. For each CASE k output EXACTLY one "
                    "line 'k: <0|1|2|3>'. Output ONLY those lines.\n\n")
            body = []
            for j, r in enumerate(batch):
                k = bi * args.batch + j
                p = re.sub(r"End 'ANSWER:.*?'\.?\s*$", "", r["prompt"]).strip()
                body.append(f"=== CASE {k} ===\n{p}\n")
                man.append({"k": k, "route": r["route"], "correct": r["correct"]})
            open(os.path.join(IO, f"b{bi}.prompt.txt"), "w", encoding="utf-8").write(head + "\n".join(body) + f"\n\nOutput the {len(batch)} lines now.")
        json.dump({"manifest": man, "nbatch": len(batches)}, open(os.path.join(IO, "manifest.json"), "w"))
        print(" ".join(f"b{i}" for i in range(len(batches))))
        return

    # score
    man = json.load(open(os.path.join(IO, "manifest.json")))["manifest"]
    by_k = {m["k"]: m for m in man}
    choice = {}
    for bi in range(len(batches)):
        fn = os.path.join(IO, f"b{bi}.resp.txt")
        if not os.path.exists(fn): continue
        for ln in open(fn, encoding="utf-8", errors="replace").read().splitlines():
            mt = re.match(r"\s*(?:CASE\s*)?(\d+)\s*[:\-]\s*([0123])", ln)
            if mt: choice[int(mt.group(1))] = int(mt.group(2))
    per = {rt: [0, 0] for rt in NOVEL}; unp = 0
    for k, m in by_k.items():
        c = choice.get(k)
        per[m["route"]][1] += 1
        if c is None: unp += 1
        elif m["correct"][c]: per[m["route"]][0] += 1
    acc = {rt: (per[rt][0] / per[rt][1] if per[rt][1] else 0.0) for rt in NOVEL}
    agg = float(sum(per[rt][0] for rt in NOVEL) / max(1, sum(per[rt][1] for rt in NOVEL)))
    rep = {"frontier_model": "gpt-5.5 (codex)", "per_route": acc, "aggregate_novel": agg,
           "n_scored": sum(per[rt][1] for rt in NOVEL), "unparsed": unp,
           "baselines_from_qwen_iclfb_run": {"gbdt_hist": 0.318, "bandit": 0.318, "random": 0.313,
                                             "conf_rule": 0.295, "llm_qwen_icl": 0.295, "oracle": 0.644}}
    json.dump(rep, open(args.out, "w"), indent=2)
    print("\n=== FRONTIER (GPT-5.5 + ICL) on zero-shot novel routes ===")
    for rt in NOVEL: print(f"  {rt:18s} {acc[rt]:.3f}")
    print(f"  AGGREGATE novel = {agg:.3f}  (scored {rep['n_scored']}, unparsed {unp})")
    print(f"  vs gbdt_hist 0.318 | bandit 0.318 | conf_rule/Qwen+ICL 0.295 | oracle 0.644")
    print(f"[WIN if] frontier+ICL > gbdt_hist 0.318")


if __name__ == "__main__":
    main()
