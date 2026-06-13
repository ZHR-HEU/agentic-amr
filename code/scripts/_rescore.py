import json, os, re, sys, random
wd = sys.argv[1]  # /tmp/agree windows path
TOOLS = ["check_acc", "detect_drift", "confusion_probe", "openset_reject", "adapt", "allocate_budget"]
def f1(p, g):
    p, g = set(p), set(g); tp = len(p & g); pr = tp/len(p) if p else 0; rc = tp/len(g) if g else 0
    return 2*pr*rc/(pr+rc) if pr+rc else 0
d = json.load(open(os.path.join(wd, "q8x.json")))
nov = [r for r in d["dump"] if r["tier"] == "novel"]; n = len(nov)
# --- (A) gap vs pre-specified single baseline router_char, bootstrap CI ---
llm = [f1(r["llm"], r["gt"]) for r in nov]
rch = [f1(r["router_char"], r["gt"]) for r in nov]
rng = random.Random(0); gaps = []
for _ in range(8000):
    ix = [rng.randrange(n) for _ in range(n)]
    gaps.append(sum(llm[i] for i in ix)/n - sum(rch[i] for i in ix)/n)
gaps.sort()
print("PRIMARY baseline = router_char (best single achievable router).")
print("  agent %.3f vs router_char %.3f ; gap %.3f  95%%CI [%.3f, %.3f]" % (
    sum(llm)/n, sum(rch)/n, sum(llm)/n-sum(rch)/n, gaps[200], gaps[7800]))
# --- (B) recall(missing-tool) vs precision(spurious-tool), per consequential tool ---
def rates(method, tool):
    miss = fp = need = notneed = 0
    for r in nov:
        gt = set(r["gt"]); pr = set(r[method])
        if tool in gt:
            need += 1; miss += (tool not in pr)
        else:
            notneed += 1; fp += (tool in pr)
    return (1 - miss/need if need else float('nan'), fp/notneed if notneed else 0.0, need)
for tool in ["adapt", "openset_reject"]:
    ar, ap, nd = rates("llm", tool); rr, rp, _ = rates("router_char", tool)
    print("  %-15s agent recall=%.2f spurious=%.2f | router_char recall=%.2f spurious=%.2f (n_need=%d)" % (
        tool, ar, ap, rr, rp, nd))
# --- (C) Cohen's kappa author vs GPT-5.5 on per-(intent,tool) inclusion ---
my = json.load(open(os.path.join(wd, "my_gt.json")))
gpt = {}
for ln in open(os.path.join(wd, "gpt_labels.txt"), encoding="utf-8", errors="ignore"):
    m = re.match(r"\s*(\d+):\s*(.*)", ln)
    if m: gpt[int(m.group(1))] = {t for t in TOOLS if t in m.group(2)}
a = b = both1 = both0 = 0; agree = 0; total = 0
for i in range(len(my)):
    if i not in gpt: continue
    for t in TOOLS:
        x = t in set(my[i]); y = t in gpt[i]
        agree += (x == y); total += 1
        a += x; b += y
        both1 += (x and y); both0 += ((not x) and (not y))
po = agree/total; pa = a/total; pb = b/total
pe = pa*pb + (1-pa)*(1-pb)
kappa = (po - pe)/(1 - pe)
print("  Author vs GPT-5.5 per-(intent,tool) label: agreement=%.3f, Cohen's kappa=%.3f (n=%d decisions)" % (po, kappa, total))
