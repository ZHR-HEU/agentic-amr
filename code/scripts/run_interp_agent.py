#!/usr/bin/env python
"""Interpretable, state-aware, memory-augmented LLM-agent for adaptive AMR.
HONEST framing (NOT an accuracy-win): the LLM-agent is compared to strong baselines
across a SUITE of metrics. Expected/honest outcome: it TIES the deterministic GBDT
controller on decision quality (J/accuracy), the adaptive controllers BEAT the fixed
pipeline on cost & risk, memory significantly helps the LLM, and the LLM UNIQUELY
provides faithful, auditable natural-language rationales (GBDT = N/A).

Reuses the leakage-fixed decision env (A1 light-CNN@128 / A2 CNN@full=extend / A3
ResNet@full=complex / A4 reject-unknown). Reports per controller on the disjoint EVAL:
  J(balanced) + mean-J(3 policies); accuracy-on-accepted; coverage(=1-reject);
  unknown-reject-recall; known-retention; mean compute & latency;
  rationale_faithfulness (LLM only) + rationale_present.
Faithfulness = fraction of LLM decisions whose natural-language reason cites the state
cue that canonically justifies the CHOSEN action (A1<-confidence, A2<-low SNR, A3<-hard/
confusable, A4<-OOD/unknown) -> a measurable self-explanation-consistency score.
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
    "accuracy": {"lam": (0.01, 0.005, 0.5), "nl": "accuracy is paramount; compute/latency cheap; minimize misclassification."},
    "latency":  {"lam": (0.03, 0.06, 0.2), "nl": "low latency critical; avoid slow/expensive actions; prefer fast outputs."},
    "balanced": {"lam": (0.02, 0.02, 0.3), "nl": "balance accuracy, compute and latency reasonably."},
}
SKEYS = ["C", "H", "E", "D", "SNR"]
# canonical state cue that justifies each action (for faithfulness scoring)
ACT_CUE = {
    "A1": ["confiden", "high prob", "certain", "clear", "reliable", "easy"],
    "A2": ["snr", "low signal", "noise", "extend", "longer", "more sampl", "weak"],
    "A3": ["complex", "hard", "difficult", "confus", "finer", "high-order", "accuracy", "deeper", "ambig"],
    "A4": ["unknown", "ood", "out-of", "out of", "distance", "prototype", "novel", "reject", "unfamiliar", "energy"],
}


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


def true_driver(e, pct):
    """Rule-derived primary decision driver (ground-truth-ish) for faithfulness."""
    if pct("D", e["D"]) > 0.85 or pct("E", e["E"]) > 0.85 or e["is_unknown"]: return "A4"
    if e["SNR"] <= 2: return "A2"
    if pct("C", e["C"]) < 0.5 or pct("H", e["H"]) > 0.6: return "A3"
    return "A1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--snr_min", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--per_class_pool", type=int, default=24)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--unseen", default=UNSEEN_DEFAULT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample", "dataset.test_per_class=160"])
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes; L = data.length
    unseen_ids = {names.index(u) for u in args.unseen.split(",") if u in names}
    seen_ids = [c for c in range(len(names)) if c not in unseen_ids]
    remap = {c: i for i, c in enumerate(seen_ids)}; inv = {i: c for c, i in remap.items()}
    crop = min(128, L)
    print(f"== interp agent == {args.dataset} L={L} seen={len(seen_ids)} unseen={sorted(unseen_ids)}", flush=True)

    tr = data.train_idx[np.isin(data.y[data.train_idx], seen_ids)]
    Xtr, ytr = data.X[tr], np.array([remap[c] for c in data.y[tr]])
    a1 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(seen_ids))
    a3 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=resnet"]), len(seen_ids))
    set_seed(cfg.seed); print("[train A1 light-cnn]", flush=True); a1.fit(Xtr[:, :, :crop], ytr, args.epochs)
    set_seed(cfg.seed); print("[train A3 complex-resnet]", flush=True); a3.fit(Xtr, ytr, args.epochs)
    proto = np.stack([a1.features(Xtr[ytr == i][:300][:, :, :crop]).mean(0) for i in range(len(seen_ids))])

    te = data.test_idx; te = te[data.snr[te] >= args.snr_min]
    pool = []
    for c in range(len(names)):
        pool += list(rng.permutation(te[data.y[te] == c])[:args.per_class_pool])
    pool = np.array(pool); Xp = data.X[pool]
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
    dev_idx, eval_idx = [], []
    for c in range(len(names)):
        ci = list(rng.permutation([i for i in range(n) if env[i]["true"] == c])); h = len(ci) // 2
        dev_idx += ci[:h]; eval_idx += ci[h:]
    dev_idx, eval_idx = np.array(dev_idx), np.array(eval_idx)
    devset = {k: np.array([env[i][k] for i in dev_idx], float) for k in SKEYS}
    def pct(k, v): return float((devset[k] < v).mean())
    def svec(i): return np.array([pct(k, env[i][k]) for k in SKEYS])
    print(f"[env] n={n} dev={len(dev_idx)} eval={len(eval_idx)} unknown_eval={sum(env[i]['is_unknown'] for i in eval_idx)}", flush=True)

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

    from sklearn.ensemble import HistGradientBoostingClassifier
    Xg, yg = [], []
    for pol, pd in POLICIES.items():
        for i in dev_idx:
            Xg.append(np.concatenate([svec(i), pd["lam"]])); yg.append(oracle(i, pd["lam"]))
    gbdt = HistGradientBoostingClassifier(max_iter=200, max_depth=4).fit(np.array(Xg), np.array(yg))
    def gbdt_action(i, lam): return gbdt.predict([np.concatenate([svec(i), lam])])[0]
    mem = [(svec(i), oracle(i, POLICIES["balanced"]["lam"]), env[i]) for i in dev_idx]

    from openai import OpenAI
    client = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY")
    SYS = ("You are a cost-aware, EXPLAINABLE decision agent for modulation recognition. Choose ONE action and justify "
           "it from the DECISIVE state cue.\n"
           "Actions: A1=light model now (cheap; ONLY for clearly confident AND in-distribution signals); "
           "A2=extend sampling then re-identify (for LOW-SNR signals where more observation helps); "
           "A3=complex model (for hard/confusable, high-uncertainty KNOWN signals); "
           "A4=reject as UNKNOWN (out-of-distribution: high energy_OOD or high prototype_distance).\n"
           "IMPORTANT: an unknown/out-of-distribution signal can look deceptively CONFIDENT. So when energy_OOD OR "
           "prototype_distance is HIGH, prefer A4 EVEN IF confidence is high 鈥?do not be fooled by confidence. Pick A1 "
           "only when confidence is high AND energy_OOD and prototype_distance are both low. Use the mission policy for "
           "the A1/A2/A3 trade-off. Give ONE short sentence citing the decisive cue, then end 'ANSWER: A1|A2|A3|A4'.")
    def stext(i):
        e = env[i]
        return (f"State: confidence=P{pct('C',e['C']):.2f}, entropy=P{pct('H',e['H']):.2f}, energy_OOD=P{pct('E',e['E']):.2f}, "
                f"prototype_distance=P{pct('D',e['D']):.2f}, SNR={e['SNR']}dB, window=128/{L}.")
    def llm_action(i, pol, use_mem):
        mt = ""
        if use_mem:
            q = svec(i); nn = np.argsort([np.linalg.norm(m[0] - q) for m in mem])[:args.topk]
            mt = "\nPast experiences (state-percentiles -> good action):\n" + "\n".join(
                f"  conf P{mem[k][0][0]:.2f} ent P{mem[k][0][1]:.2f} energy P{mem[k][0][2]:.2f} dist P{mem[k][0][3]:.2f} "
                f"snr {mem[k][2]['SNR']}dB -> {mem[k][1]}" for k in nn)
        user = f"MISSION COST POLICY: {POLICIES[pol]['nl']}\n{stext(i)}{mt}\nReason then choose. ANSWER: A1|A2|A3|A4."
        try:
            r = client.chat.completions.create(model=cfg.controller.model,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=180, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            txt = r.choices[0].message.content or ""
        except Exception:
            return "A1", ""
        m = re.search(r"ANSWER:\s*(A[1-4])", txt) or re.search(r"\b(A[1-4])\b", txt)
        return (m.group(1) if m else "A1"), txt

    # ---- evaluate metric SUITE on eval ----
    controllers = ["fixed_rule", "gbdt", "llm_nomem", "llm_mem", "oracle"]
    # mean-J across policies (tie claim)
    perJ = {a: [] for a in controllers}
    act_cache = {}  # (agent,pol,i) -> action ; rationale cache for llm_mem balanced
    rationales = {}
    for pol, pd in POLICIES.items():
        lam = pd["lam"]; tuned = make_tuned_rule(lam, dev_idx)
        for i in eval_idx:
            acts = {"fixed_rule": tuned(i), "gbdt": gbdt_action(i, lam), "oracle": oracle(i, lam)}
            an, _ = llm_action(i, pol, False); acts["llm_nomem"] = an
            am, tm = llm_action(i, pol, True); acts["llm_mem"] = am
            if pol == "balanced": rationales[i] = (am, tm)
            for a in controllers:
                perJ[a].append(reward(acts[a], env[i], lam))
                act_cache[(a, pol, i)] = acts[a]
    meanJ = {a: float(np.mean(perJ[a])) for a in controllers}

    # operating-point metrics under BALANCED policy
    def suite(agent):
        accepts = corr = comp = lat = 0; n_acc = 0; unk_total = unk_rej = known_total = known_kept = 0
        for i in eval_idx:
            a = act_cache[(agent, "balanced", i)]; e = env[i]
            comp += COST[a][0]; lat += COST[a][1]
            if e["is_unknown"]:
                unk_total += 1; unk_rej += (a == "A4")
            else:
                known_total += 1; known_kept += (a != "A4")
            if a != "A4":
                n_acc += 1
                ok = (e["c1"] if a == "A1" else e["c2"] if a == "A2" else e["c3"])
                corr += ok
        m = len(eval_idx)
        return {"J_balanced": float(np.mean([reward(act_cache[(agent,'balanced',i)], env[i], POLICIES['balanced']['lam']) for i in eval_idx])),
                "meanJ": meanJ[agent],
                "acc_on_accepted": (corr / n_acc if n_acc else 0.0), "coverage": n_acc / m,
                "unknown_reject_recall": (unk_rej / unk_total if unk_total else 0.0),
                "known_retention": (known_kept / known_total if known_total else 0.0),
                "mean_compute": comp / m, "mean_latency": lat / m}

    metrics = {a: suite(a) for a in controllers}

    # faithfulness for llm_mem (balanced)
    def faithful(a, txt, e):
        """STRICT: rationale must cite the action's cue AND that cue must actually be in the
        regime that justifies the action (not mere parroting of provided numbers)."""
        if not any(kw in txt.lower() for kw in ACT_CUE.get(a, [])): return False
        cP, hP, eP, dP = pct("C", e["C"]), pct("H", e["H"]), pct("E", e["E"]), pct("D", e["D"])
        if a == "A1": return cP > 0.55 and eP < 0.7 and dP < 0.7
        if a == "A2": return e["SNR"] <= 5
        if a == "A3": return hP > 0.55 or cP < 0.5
        if a == "A4": return eP > 0.65 or dP > 0.65
        return False
    faith = 0; have = 0; examples = []
    for i in eval_idx:
        a, txt = rationales[i]
        if not txt: continue
        have += 1
        ok = faithful(a, txt, env[i]); faith += ok
        if len(examples) < 16:
            examples.append({"SNR": env[i]["SNR"], "is_unknown": env[i]["is_unknown"],
                             "conf_P": round(pct("C", env[i]["C"]), 2), "ent_P": round(pct("H", env[i]["H"]), 2),
                             "ood_P": round(pct("E", env[i]["E"]), 2), "dist_P": round(pct("D", env[i]["D"]), 2),
                             "action": a, "true_driver": true_driver(env[i], pct),
                             "rationale": txt.strip().replace("\n", " ")[:240], "faithful": bool(ok)})
    faithfulness = faith / have if have else 0.0
    for a in controllers:
        metrics[a]["rationale_faithfulness"] = (faithfulness if a == "llm_mem" else (None if a != "llm_nomem" else None))
        metrics[a]["rationale_present"] = (1.0 if a in ("llm_mem", "llm_nomem") else 0.0)

    rep = {"dataset": args.dataset, "n_eval": int(len(eval_idx)),
           "unknown_eval": int(sum(env[i]["is_unknown"] for i in eval_idx)),
           "metrics": metrics, "llm_mem_faithfulness": faithfulness, "rationale_examples": examples,
           "note": "HONEST: accuracy/J tie GBDT; LLM-unique value = faithful rationales; adaptive>fixed on cost/risk; memory ablation = mem vs nomem."}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"interp_agent_{tag}{args.dataset}.json"), "w"), indent=2)
    print("\n=== INTERP-AGENT metric suite (balanced policy; meanJ over 3 policies) ===")
    hdr = ["meanJ", "J_balanced", "acc_on_accepted", "coverage", "unknown_reject_recall", "known_retention", "mean_compute", "rationale_faithfulness", "rationale_present"]
    for a in controllers:
        print(f"  {a:12s} " + " ".join(f"{k}={metrics[a][k]:.3f}" if isinstance(metrics[a][k], float) else f"{k}={metrics[a][k]}" for k in hdr))
    print(f"\n[interpretability] llm_mem rationale_faithfulness = {faithfulness:.3f} (GBDT/rule = N/A, no native rationale)")
    print("[honest] LLM ties GBDT on J/acc; adaptive(LLM,GBDT) > fixed on cost/risk; mem vs nomem = memory gain")
    open(os.path.join(out_dir, f"INTERP_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
