#!/usr/bin/env python
"""Make-or-break decision gate (Innovation 2), leakage-fixed + strong baselines.

Decision env (RML2018): per sample 4 actions with REALIZED outcomes/costs:
  A1 light-CNN@short(128) | A2 light-CNN@full(=extend sampling,+lat) |
  A3 complex ResNet@full(+compute+lat) | A4 reject-unknown(correct iff held-out class).
State s=[C,H,E,D,SNR]; reward J=correct - l1*compute - l2*latency - l3*error.

STRICT dev/eval split: prototypes/pctl/memory/rule-tuning/GBDT fit on DEV only;
all agents evaluated once on disjoint EVAL. 3 NL cost policies (accuracy/latency/balanced).
Agents: oracle | fixed_rule(tuned on balanced) | pertuned_rule(per-policy) |
gbdt(policy-conditioned, lambda as input) | llm_nomem | llm_mem(memory from DEV).
Pre-registered WIN: llm_mem mean-J >= gbdt AND >= pertuned_rule (bootstrap CI on the
gap excludes <0) AND >> fixed_rule AND >= llm_nomem. (Honest expectation: gbdt ties/beats LLM.)
"""
import argparse, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier

UNSEEN_DEFAULT = "FM,GMSK,OQPSK,OOK"
ACTIONS = ["A1", "A2", "A3", "A4"]
COST = {"A1": (1.0, 1.0), "A2": (1.5, 4.0), "A3": (8.0, 6.0), "A4": (1.0, 1.0)}
POLICIES = {
    "accuracy": {"lam": (0.01, 0.005, 0.5),
                 "nl": "Mission: accuracy is paramount; compute and latency are cheap; minimize misclassification at almost any cost."},
    "latency":  {"lam": (0.03, 0.06, 0.2),
                 "nl": "Mission: low latency is critical (real-time); avoid slow/expensive actions; prefer fast outputs even at some accuracy cost."},
    "balanced": {"lam": (0.02, 0.02, 0.3),
                 "nl": "Mission: balance accuracy, compute and latency reasonably."},
}
SKEYS = ["C", "H", "E", "D", "SNR"]


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def reward(a, e, lam):
    l1, l2, l3 = lam; comp, lat = COST[a]
    if a == "A4": correct = e["is_unknown"]
    elif a == "A1": correct = e["c1"] and not e["is_unknown"]
    elif a == "A2": correct = e["c2"] and not e["is_unknown"]
    else: correct = e["c3"] and not e["is_unknown"]
    return (1.0 if correct else 0.0) - l1 * comp - l2 * lat - l3 * (0.0 if correct else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--snr_min", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--per_class_pool", type=int, default=24)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--unseen", default=UNSEEN_DEFAULT)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample",
                                     "dataset.test_per_class=160"])
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes; L = data.length
    unseen_ids = {names.index(u) for u in args.unseen.split(",") if u in names}
    seen_ids = [c for c in range(len(names)) if c not in unseen_ids]
    remap = {c: i for i, c in enumerate(seen_ids)}; inv = {i: c for c, i in remap.items()}
    crop = min(128, L)
    print(f"== decision gate v2 == {args.dataset} L={L} seen={len(seen_ids)} unseen={sorted(unseen_ids)}", flush=True)

    # train A1 (cnn, short) + A3 (resnet, full) on SEEN training data
    tr = data.train_idx[np.isin(data.y[data.train_idx], seen_ids)]
    Xtr, ytr = data.X[tr], np.array([remap[c] for c in data.y[tr]])
    a1 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(seen_ids))
    a3 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=resnet"]), len(seen_ids))
    set_seed(cfg.seed); print("[train A1]", flush=True); a1.fit(Xtr[:, :, :crop], ytr, args.epochs)
    set_seed(cfg.seed); print("[train A3]", flush=True); a3.fit(Xtr, ytr, args.epochs)
    proto = np.stack([a1.features(Xtr[ytr == i][:300][:, :, :crop]).mean(0) for i in range(len(seen_ids))])

    # build env on test pool (seen + unknown), snr>=snr_min, stratified
    te = data.test_idx
    te = te[data.snr[te] >= args.snr_min]
    pool = []
    for c in range(len(names)):
        idx = te[data.y[te] == c]
        pool += list(rng.permutation(idx)[:args.per_class_pool])
    pool = np.array(pool)
    Xp = data.X[pool]
    p1 = a1.predict_proba(Xp[:, :, :crop]); ll = a1._logits(Xp[:, :, :crop]).numpy(); f1 = a1.features(Xp[:, :, :crop])
    p2 = a1.predict_proba(Xp); p3 = a3.predict_proba(Xp)
    env = []
    for j, gi in enumerate(pool):
        true = int(data.y[gi]); unk = true in unseen_ids
        env.append({"true": true, "is_unknown": unk,
                    "c1": (inv[int(p1[j].argmax())] == true) and not unk,
                    "c2": (inv[int(p2[j].argmax())] == true) and not unk,
                    "c3": (inv[int(p3[j].argmax())] == true) and not unk,
                    "C": float(p1[j].max()), "H": float(-(p1[j] * np.log(p1[j] + 1e-12)).sum()),
                    "E": float(-(np.log(np.exp(ll[j] - ll[j].max()).sum()) + ll[j].max())),
                    "D": float(np.linalg.norm(proto - f1[j], axis=1).min()), "SNR": int(data.snr[gi])})
    n = len(env)
    # stratified dev/eval split (50/50 by class)
    dev_idx, eval_idx = [], []
    for c in range(len(names)):
        ci = [i for i in range(n) if env[i]["true"] == c]
        ci = list(rng.permutation(ci)); h = len(ci) // 2
        dev_idx += ci[:h]; eval_idx += ci[h:]
    dev_idx, eval_idx = np.array(dev_idx), np.array(eval_idx)
    devset = {k: np.array([env[i][k] for i in dev_idx], float) for k in SKEYS}
    def pct(k, v): return float((devset[k] < v).mean())
    def svec(i): return np.array([pct(k, env[i][k]) for k in SKEYS])
    print(f"[env] n={n} dev={len(dev_idx)} eval={len(eval_idx)} unknown_eval="
          f"{sum(env[i]['is_unknown'] for i in eval_idx)}", flush=True)

    def oracle(i, lam): return max(ACTIONS, key=lambda a: reward(a, env[i], lam))

    def make_tuned_rule(lam, idxs):
        best, bJ = None, -1e9
        for ct in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for ot in [0.8, 0.9, 0.95]:
                for esc in ["A3", "A2", "A1"]:
                    def f(i, ct=ct, ot=ot, esc=esc):
                        e = env[i]
                        if pct("D", e["D"]) > ot or pct("E", e["E"]) > ot: return "A4"
                        if pct("C", e["C"]) > ct: return "A1"
                        return esc
                    J = np.mean([reward(f(i), env[i], lam) for i in idxs])
                    if J > bJ: bJ, best = J, f
        return best

    # policy-conditioned GBDT trained on DEV across all policies (lambda as input)
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xg, yg = [], []
    for pol, pd in POLICIES.items():
        lam = pd["lam"]
        for i in dev_idx:
            Xg.append(np.concatenate([svec(i), lam])); yg.append(oracle(i, lam))
    gbdt = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.1)
    gbdt.fit(np.array(Xg), np.array(yg))
    def gbdt_action(i, lam): return gbdt.predict([np.concatenate([svec(i), lam])])[0]

    # memory bank from DEV (state, action-best-under-balanced, env)
    mem = [(svec(i), oracle(i, POLICIES["balanced"]["lam"]), env[i]) for i in dev_idx]

    from openai import OpenAI
    client = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY")
    SYS = ("You are a cost-aware decision agent for modulation recognition. Given the signal STATE and a "
           "MISSION COST POLICY, choose ONE action maximizing the mission objective. A1=light model now "
           "(compute1,lat1); A2=extend sampling+re-ID (compute1.5,lat4, helps low SNR); A3=complex model "
           "(compute8,lat6, higher accuracy); A4=reject UNKNOWN (high prototype-distance/energy). End 'ANSWER: A1|A2|A3|A4'.")
    def stext(i):
        e = env[i]
        return (f"State: confidence=P{pct('C',e['C']):.2f}, entropy=P{pct('H',e['H']):.2f}, "
                f"energy_OOD=P{pct('E',e['E']):.2f}, prototype_distance=P{pct('D',e['D']):.2f}, SNR={e['SNR']}dB, window=128/{L}.")
    def llm_action(i, pol, use_mem):
        mt = ""
        if use_mem:
            q = svec(i); d = [np.linalg.norm(m[0] - q) for m in mem]; nn = np.argsort(d)[:args.topk]
            mt = "\nPast experiences (state-percentiles -> good action):\n" + "\n".join(
                f"  conf P{mem[k][0][0]:.2f} ent P{mem[k][0][1]:.2f} energy P{mem[k][0][2]:.2f} "
                f"dist P{mem[k][0][3]:.2f} snr {mem[k][2]['SNR']}dB -> {mem[k][1]}" for k in nn)
        user = f"MISSION COST POLICY: {POLICIES[pol]['nl']}\n{stext(i)}{mt}\nChoose the action. ANSWER: A1|A2|A3|A4."
        try:
            r = client.chat.completions.create(model=cfg.controller.model,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=180, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            txt = r.choices[0].message.content or ""
        except Exception:
            return "A1"
        m = re.search(r"ANSWER:\s*(A[1-4])", txt) or re.search(r"\b(A[1-4])\b", txt)
        return m.group(1) if m else "A1"

    fixed_rule = make_tuned_rule(POLICIES["balanced"]["lam"], dev_idx)
    agents = ["fixed_rule", "pertuned_rule", "gbdt", "llm_nomem", "llm_mem"]
    report = {"dataset": args.dataset, "n_eval": int(len(eval_idx)), "policies": {}, "perJ": {}}
    perJ = {a: {} for a in agents + ["oracle"]}      # agent -> policy -> list of per-sample J on eval
    for pol, pd in POLICIES.items():
        lam = pd["lam"]
        tuned = make_tuned_rule(lam, dev_idx)
        actfns = {"fixed_rule": lambda i: fixed_rule(i), "pertuned_rule": lambda i: tuned(i),
                  "gbdt": lambda i: gbdt_action(i, lam),
                  "llm_nomem": lambda i: llm_action(i, pol, False),
                  "llm_mem": lambda i: llm_action(i, pol, True)}
        perJ["oracle"][pol] = [reward(oracle(i, lam), env[i], lam) for i in eval_idx]
        res = {"oracle_J": float(np.mean(perJ["oracle"][pol]))}
        for a in agents:
            acts = [actfns[a](i) for i in eval_idx]
            js = [reward(acts[k], env[i], lam) for k, i in enumerate(eval_idx)]
            perJ[a][pol] = js
            res[a] = {"J": float(np.mean(js)),
                      "dec_acc": float(np.mean([acts[k] == oracle(i, lam) for k, i in enumerate(eval_idx)]))}
        report["policies"][pol] = res
        print(f"[{pol}] oracle={res['oracle_J']:.3f} fixed={res['fixed_rule']['J']:.3f} "
              f"pertuned={res['pertuned_rule']['J']:.3f} gbdt={res['gbdt']['J']:.3f} "
              f"nomem={res['llm_nomem']['J']:.3f} mem={res['llm_mem']['J']:.3f}", flush=True)

    # mean-J across policies + bootstrap CI of key gaps (mem - X), pooled per-sample over policies
    def pooled(a): return np.concatenate([perJ[a][p] for p in POLICIES])
    meanJ = {a: float(pooled(a).mean()) for a in agents + ["oracle"]}
    def boot_gap(a, b, B=2000):
        da = pooled(a) - pooled(b); m = len(da)
        bs = [da[rng.integers(0, m, m)].mean() for _ in range(B)]
        return float(np.mean(da)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    gaps = {f"mem_vs_{b}": boot_gap("llm_mem", b) for b in ["gbdt", "pertuned_rule", "fixed_rule", "llm_nomem"]}
    report["mean_J"] = meanJ; report["gaps(mem-x) mean[ci95]"] = gaps
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"decision_gate_{args.dataset}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[mean-J] oracle={meanJ['oracle']:.3f} fixed={meanJ['fixed_rule']:.3f} pertuned={meanJ['pertuned_rule']:.3f} "
          f"gbdt={meanJ['gbdt']:.3f} llm_nomem={meanJ['llm_nomem']:.3f} llm_mem={meanJ['llm_mem']:.3f}", flush=True)
    for k, (m, lo, hi) in gaps.items():
        print(f"[gap {k}] {m:+.3f} (95% CI {lo:+.3f},{hi:+.3f})", flush=True)
    open(os.path.join(out_dir, f"DECGATE_DONE_{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
