import json, os, re, sys
wd = sys.argv[1]
TOOLS = ["check_acc", "detect_drift", "confusion_probe", "openset_reject", "adapt", "allocate_budget"]
def f1(p, g):
    p, g = set(p), set(g); tp = len(p & g); pr = tp/len(p) if p else 0; rc = tp/len(g) if g else 0
    return 2*pr*rc/(pr+rc) if pr+rc else 0
my = [set(x) for x in json.load(open(os.path.join(wd, "my_gt.json")))]
gpt = {}
for ln in open(os.path.join(wd, "gpt_labels.txt"), encoding="utf-8", errors="ignore"):
    m = re.match(r"\s*(\d+):\s*(.*)", ln)
    if m: gpt[int(m.group(1))] = {t for t in TOOLS if t in m.group(2)}
gem = {int(k): set(v) for k, v in json.load(open(os.path.join(wd, "gemma_labels.json"))).items()}
n = len(my); idxs = [i for i in range(n) if i in gpt and i in gem]

def kappa(a, b):  # Cohen on per-(intent,tool) binary
    po = pa = pb = tot = 0
    for i in idxs:
        for t in TOOLS:
            x = t in a[i]; y = t in b[i]; po += (x == y); pa += x; pb += y; tot += 1
    po /= tot; pa /= tot; pb /= tot; pe = pa*pb + (1-pa)*(1-pb)
    return (po - pe)/(1 - pe)
A = {i: my[i] for i in idxs}
print("Pairwise Cohen's kappa (per-(intent,tool) binary, n=%d intents):" % len(idxs))
print("  author-GPT5.5=%.3f  author-Gemma=%.3f  GPT5.5-Gemma=%.3f" % (kappa(A, gpt), kappa(A, gem), kappa(gpt, gem)))
# Fleiss' kappa: 3 raters, each (intent,tool) item -> #raters saying "include"
items = []
for i in idxs:
    for t in TOOLS:
        items.append(sum([t in A[i], t in gpt[i], t in gem[i]]))
N = len(items); nr = 3
p_inc = sum(items) / (N * nr); p_exc = 1 - p_inc
Pe = p_inc**2 + p_exc**2
Pbar = sum((c**2 + (nr-c)**2 - nr) / (nr*(nr-1)) for c in items) / N
fleiss = (Pbar - Pe) / (1 - Pe)
print("  Fleiss' kappa (3 annotators, %d binary items) = %.3f" % (N, fleiss))
# agent F1 vs Gemma labels
d = json.load(open(os.path.join(wd, "q8x.json"))); nov = [r for r in d["dump"] if r["tier"] == "novel"]
routers = ["router_nn", "router_char", "router_ml", "router_ml_char"]
llm = [f1(nov[i]["llm"], gem[i]) for i in idxs]
br = [max(f1(nov[i][rt], gem[i]) for rt in routers) for i in idxs]
print("Agent vs Gemma labels: agent F1=%.3f  best-router F1=%.3f  gap=%.3f" % (
    sum(llm)/len(llm), sum(br)/len(br), sum(llm)/len(llm)-sum(br)/len(br)))
