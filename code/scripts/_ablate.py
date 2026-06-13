import sys, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_agent_intent as R
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
TOOLS = R.TOOLS
def f1(p, g):
    p, g = set(p), set(g); tp = len(p & g); pr = tp/len(p) if p else 0; rc = tp/len(g) if g else 0
    return 2*pr*rc/(pr+rc) if pr+rc else 0
base = [(b[0], set(b[1])) for b in R.BANK if b[2] == "base"]
novel = [(b[0], set(b[1])) for b in R.BANK if b[2] == "novel"]
rng = np.random.default_rng(0); idx = rng.permutation(len(novel))
def train_router(train):
    X = [t[0] for t in train]; cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(X); C = cvec.transform(X)
    ml = {}
    for t in TOOLS:
        y = [1 if t in d[1] else 0 for d in train]
        ml[t] = LogisticRegression(max_iter=1000).fit(C, y) if 0 < sum(y) < len(y) else None
    def pred(intent):
        v = cvec.transform([intent]); out = {t for t in TOOLS if ml[t] is not None and ml[t].predict(v)[0] == 1}
        if out: return out
        k = int((C @ v.T).toarray().ravel().argmax()); return set(train[k][1])
    return pred
half = len(novel) // 2
folds = [(idx[:half], idx[half:]), (idx[half:], idx[:half])]
fs = []
for tr, te in folds:
    train = base + [novel[i] for i in tr]; pred = train_router(train)
    fs += [f1(pred(novel[i][0]), novel[i][1]) for i in te]
print("COMPOUND-TRAINED router (base + half novel) novel-CV F1 = %.3f (n=%d)" % (float(np.mean(fs)), len(fs)))
print("  vs base-only router 0.567 | agent(Qwen) 0.895")
