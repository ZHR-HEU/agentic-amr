#!/usr/bin/env python
"""FROZEN-POLICY TRANSFER gate (Codex's #1 experiment) 鈥?the decisive answer to
"why not just a deterministic rule?". Deterministic controllers (tuned threshold,
GBDT, CART) are TUNED on TRAIN drift-schedules + TRAIN unknown families, then FROZEN
and evaluated on HELD-OUT unknown families + NEW drift schedules. The LLM agent is
frozen too (never tuned). If the tuned controllers TRANSFER poorly (degrade on heldout
conditions) while the LLM's state-card reasoning holds, that justifies the LLM.

Metrics on the TEST (held-out) streams: overall-J, drift+novelty-seg J, regret-to-
clairvoyant-oracle, unknown reject recall AND precision, false-reject of low-SNR knowns,
adaptation lag, action-switch rate, + TRANSFER GAP (train-J - test-J per tuned method).
Baselines: fixed_rule(tuned), gbdt(tuned), cart(tuned), online_rule, bandit(online),
oracle(per-step), oracle_tuned_on_test(upper bound). LLM via openai|hf.
"""
import argparse, json, os, re, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from amrl.config import load_config
from amrl.data import load_dataset
from amrl.model import Classifier

ACTIONS = ["A1", "A2", "A3", "A4"]
COST = {"A1": (1.0, 1.0), "A2": (1.5, 4.0), "A3": (8.0, 6.0), "A4": (1.0, 1.0)}
LAM = (0.02, 0.02, 0.3)
SKEYS = ["C", "H", "E", "D", "SNR"]
ACT_CUE = {"A1": ["confiden", "clear", "easy", "reliable", "in-distribution", "in distribution"],
           "A2": ["snr", "noise", "weak", "extend", "longer", "low signal", "dropping", "degrad"],
           "A3": ["complex", "hard", "confus", "uncertain", "ambig", "finer", "high-order"],
           "A4": ["unknown", "ood", "out-of", "out of", "distance", "prototype", "novel", "reject", "unfamiliar", "energy"]}
# drift schedules: TRAIN ones tuned on; TEST ones held out (different temporal profiles)
def snr_at(sched, t, S):
    seg, u = t // S, (t % S) / S
    if sched == "abrupt":   return [16, -10, 14, -8][seg % 4]
    if sched == "ramp":     return [16, 16 - 24 * u, -8, -8 + 22 * u][seg]
    if sched == "recurrent":return 4 + 12 * np.sin(2 * np.pi * t / (S))
    if sched == "gradcrash":return [14, 14 - 8 * u, 6 - 22 * u, -16 + 30 * u][seg]
    return 0.0
TRAIN_SCHED = ["abrupt", "ramp"]; TEST_SCHED = ["recurrent", "gradcrash"]


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def reward(a, e):
    l1, l2, l3 = LAM; comp, lat = COST[a]
    if a == "A4": correct = e["is_unknown"]
    elif a == "A1": correct = e["c1"] and not e["is_unknown"]
    elif a == "A2": correct = e["c2"] and not e["is_unknown"]
    else: correct = e["c3"] and not e["is_unknown"]
    return (1.0 if correct else 0.0) - l1 * comp - l2 * lat - l3 * (0.0 if correct else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seg", type=int, default=40)
    ap.add_argument("--n_seeds", type=int, default=8, help="# stream seeds (train-once, eval-many for CIs)")
    ap.add_argument("--win", type=int, default=40)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--unseen_train", default="")   # auto per dataset if empty
    ap.add_argument("--unseen_test", default="")
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--backend", default="openai", choices=["openai", "hf", "none"])
    ap.add_argument("--hf_model_path", default="models/gemma-4-12B-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reject_mode", default="conservative", choices=["aggressive", "moderate", "conservative"],
                    help="NL-tuned novelty-rejection strictness (operating point on the precision-recall frontier)")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    ov = [f"dataset.name={args.dataset}", "dataset.normalize=per_sample", "dataset.test_per_class=200"]
    if args.seed >= 0: ov.append(f"seed={args.seed}")
    cfg = load_config(args.config, ov)
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes; L = data.length
    # default held-out unknown families per dataset (TRAIN vs TEST disjoint)
    if not args.unseen_train:
        args.unseen_train = "OOK,OQPSK" if args.dataset == "rml2018" else "WBFM,GFSK"
    if not args.unseen_test:
        args.unseen_test = "FM,GMSK" if args.dataset == "rml2018" else "PAM4,CPFSK"
    U_tr = [names.index(u) for u in args.unseen_train.split(",") if u in names]
    U_te = [names.index(u) for u in args.unseen_test.split(",") if u in names]
    seen_ids = [c for c in range(len(names)) if c not in set(U_tr) | set(U_te)]
    remap = {c: i for i, c in enumerate(seen_ids)}; inv = {i: c for c, i in remap.items()}
    crop = args.crop if args.crop > 0 else min(128, L)
    print(f"== TRANSFER == {args.dataset} crop={crop}/{L} seen={len(seen_ids)} U_train={sorted(U_tr)} U_test={sorted(U_te)}", flush=True)

    tr = data.train_idx[np.isin(data.y[data.train_idx], seen_ids)]
    Xtr, ytr = data.X[tr], np.array([remap[c] for c in data.y[tr]])
    a1 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(seen_ids))
    a3 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=resnet"]), len(seen_ids))
    set_seed(cfg.seed); print("[train A1]", flush=True); a1.fit(Xtr[:, :, :crop], ytr, args.epochs)
    set_seed(cfg.seed); print("[train A3]", flush=True); a3.fit(Xtr, ytr, args.epochs)
    proto = np.stack([a1.features(Xtr[ytr == i][:300][:, :, :crop]).mean(0) for i in range(len(seen_ids))])
    te_pool = data.test_idx

    def make_env(gi):
        x = data.X[gi:gi + 1]; true = int(data.y[gi]); unk = true not in seen_ids
        p1 = a1.predict_proba(x[:, :, :crop])[0]; lg = a1._logits(x[:, :, :crop]).numpy()[0]; f1 = a1.features(x[:, :, :crop])[0]
        p2 = a1.predict_proba(x)[0]; p3 = a3.predict_proba(x)[0]
        return {"true": true, "is_unknown": unk,
                "c1": (inv[int(p1.argmax())] == true) and not unk, "c2": (inv[int(p2.argmax())] == true) and not unk,
                "c3": (inv[int(p3.argmax())] == true) and not unk,
                "C": float(p1.max()), "H": float(-(p1 * np.log(p1 + 1e-12)).sum()),
                "E": float(-(np.log(np.exp(lg - lg.max()).sum()) + lg.max())),
                "D": float(np.linalg.norm(proto - f1, axis=1).min()), "SNR": int(data.snr[gi])}

    def sample_idx(snr_t, unknown_pool):
        ids = unknown_pool if unknown_pool is not None else seen_ids
        cand = te_pool[np.isin(data.y[te_pool], list(ids)) & (np.abs(data.snr[te_pool] - snr_t) <= 2.5)]
        if len(cand) == 0: cand = te_pool[np.isin(data.y[te_pool], list(ids))]
        return int(rng.choice(cand))

    def gen_stream(sched, unk_pool, seed):
        rg = np.random.default_rng(seed); S = args.seg; T = 4 * S; rows = []
        for t in range(T):
            snr = float(snr_at(sched, t, S)); unk = (rg.random() < 0.4) and (snr < 6)
            rows.append(make_env(sample_idx(snr, unk_pool if unk else None)))
        return rows

    calib = [make_env(int(gi)) for c in seen_ids for gi in rng.permutation(te_pool[data.y[te_pool] == c])[:10]]
    calset = {k: np.array([e[k] for e in calib], float) for k in SKEYS}
    def pct(k, v): return float((calset[k] < v).mean())
    def svec(e): return np.array([pct(k, e[k]) for k in SKEYS])

    # TRAIN streams (train schedules + train unknown family) to TUNE the deterministic controllers
    train_rows = []
    for sd in TRAIN_SCHED:
        for r in range(2): train_rows += gen_stream(sd, U_tr, cfg.seed + 700 + hash(sd) % 100 + r)
    Xt = np.array([svec(e) for e in train_rows]); yt = np.array([max(ACTIONS, key=lambda a: reward(a, e)) for e in train_rows])
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    gbdt = HistGradientBoostingClassifier(max_iter=200, max_depth=4).fit(Xt, yt)
    cart = DecisionTreeClassifier(max_depth=4).fit(Xt, yt)
    def tune_rule(rows):
        best, bJ = None, -1e9
        for ot in [0.75, 0.8, 0.85, 0.9]:
            for ct in [0.4, 0.5, 0.6, 0.7]:
                for esc in ["A3", "A2"]:
                    def f(e, ot=ot, ct=ct, esc=esc):
                        if pct("D", e["D"]) > ot or pct("E", e["E"]) > ot: return "A4"
                        if pct("C", e["C"]) > ct: return "A1"
                        return esc
                    J = np.mean([reward(f(e), e) for e in rows])
                    if J > bJ: bJ, best = J, f
        return best
    fixed_rule = tune_rule(train_rows)
    trainJ = {"fixed_rule": np.mean([reward(fixed_rule(e), e) for e in train_rows]),
              "gbdt": np.mean([reward(str(gbdt.predict([svec(e)])[0]), e) for e in train_rows]),
              "cart": np.mean([reward(str(cart.predict([svec(e)])[0]), e) for e in train_rows])}

    def online_rule(e, win):
        if len(win) < 8: return fixed_rule(e)
        def opc(k, v): a = np.array([w[k] for w in win]); return float((a < v).mean())
        if opc("D", e["D"]) > 0.85 or opc("E", e["E"]) > 0.85: return "A4"
        if opc("C", e["C"]) > 0.6: return "A1"
        return "A3"

    # LLM 鈥?the novelty-rejection rule is a NATURAL-LANGUAGE-TUNABLE operating point on the
    # precision-recall frontier. Three strictness levels are selected by --reject_mode; the
    # ONLY text that differs between them is REJECT_RULE, so the controllability curve is clean.
    REJECT_RULE = {
        "aggressive": (
            "REJECTION RULE: choose A4 whenever the signal looks out-of-distribution 鈥?energy_OOD OR "
            "prototype_distance is HIGH (>~P0.70), or confidence is very low with high entropy. When in "
            "doubt about novelty, prefer A4 over passing an unknown to the classifier."),
        "moderate": (
            "REJECTION RULE: choose A4 when out-of-distribution evidence is CLEAR 鈥?either (a) energy_OOD "
            "AND prototype_distance are BOTH at least moderately high (>~P0.65), OR (b) one of them is VERY "
            "high (>~P0.90). A weak but in-distribution low-SNR signal has high entropy yet keeps energy_OOD/"
            "distance near typical (<~P0.60) 鈫?use A2 (extend), NOT A4. Do not reject for low confidence/SNR alone."),
        "conservative": (
            "CALIBRATED REJECTION RULE: choose A4 ONLY when BOTH energy_OOD AND prototype_distance are clearly "
            "HIGH (>~P0.80) 鈥?that signature means out-of-distribution. LOW SNR ALONE is NOT a reason to reject: "
            "a weak but in-distribution signal has high entropy yet only MODERATE energy_OOD/distance, so use A2 "
            "(extend), NOT A4. Do not over-reject known signals just because confidence is low."),
    }[args.reject_mode]
    SYS = ("You are an EXPLAINABLE, ADAPTIVE cost-aware decision agent for online modulation recognition under "
           "NON-STATIONARY drift with possibly UNKNOWN signals. A1=light model (confident,in-distribution); A2=extend "
           "sampling (for LOW-SNR but in-distribution signals); A3=complex model (hard/uncertain known); A4=reject as "
           "UNKNOWN. " + REJECT_RULE + " Use RECENT EXPERIENCES to adapt. One "
           "short sentence citing the decisive cue, then 'ANSWER: A1|A2|A3|A4'.")
    def build_user(e, mr, rs):
        mt = "" if not mr else "\nRecent experiences (percentiles -> good action):\n" + "\n".join(
            f"  conf P{m[0][0]:.2f} ent P{m[0][1]:.2f} energy P{m[0][2]:.2f} dist P{m[0][3]:.2f} snr {m[1]}dB -> {m[2]}" for m in mr)
        tr_ = "" if len(rs) < 4 else f"\nRecent SNR trend: {', '.join(f'{s:.0f}' for s in rs[-6:])} dB."
        return (f"State: confidence=P{pct('C',e['C']):.2f}, entropy=P{pct('H',e['H']):.2f}, energy_OOD=P{pct('E',e['E']):.2f}, "
                f"prototype_distance=P{pct('D',e['D']):.2f}, SNR={e['SNR']}dB.{tr_}{mt}\nReason then choose. ANSWER: A1|A2|A3|A4.")
    def parse_act(t):
        m = re.search(r"ANSWER:\s*(A[1-4])", t) or re.search(r"\b(A[1-4])\b", t); return m.group(1) if m else "A1"
    chat = None
    if args.backend == "openai":
        from openai import OpenAI
        cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
        def chat(e, mr, rs):
            try:
                r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": build_user(e, mr, rs)}],
                    temperature=0.0, max_tokens=160, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                return parse_act(r.choices[0].message.content or ""), (r.choices[0].message.content or "")
            except Exception: return "A1", ""
    elif args.backend == "hf":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.hf_model_path)
        mdl = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16, device_map=args.device).eval()
        model = os.path.basename(args.hf_model_path.rstrip("/"))
        def chat(e, mr, rs):
            enc = tok.apply_chat_template([{"role": "user", "content": SYS + "\n\n" + build_user(e, mr, rs)}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            with torch.no_grad(): out = mdl.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True); return parse_act(txt), txt
    else: model = "none"

    methods = ["oracle", "oracle_tuned_test", "fixed_rule", "online_rule", "gbdt", "cart", "bandit"] + (["llm_online_mem"] if chat else [])
    METRICS = ["overallJ", "regret", "unk_reject_recall", "unk_reject_prec", "false_reject_lowSNRknown", "switch_rate"]
    # train/tune was done ONCE above (a1/a3/gbdt/cart/fixed_rule). Now loop over STREAM seeds and aggregate.
    perseed = {m: {k: [] for k in METRICS} for m in methods}
    for si in range(args.n_seeds):
        sm = {m: collections.defaultdict(list) for m in methods}
        for sd in TEST_SCHED:
            rows = gen_stream(sd, U_te, 10000 + si * 131 + (abs(hash(sd)) % 50))
            q = np.zeros(4); ncq = np.zeros(4) + 1e-6
            onlinemem = collections.deque(maxlen=args.win); onwin = collections.deque(maxlen=args.win); rs = []
            otr = tune_rule(rows)
            acts = {m: [] for m in methods}
            for e in rows:
                rs.append(e["SNR"]); best = max(ACTIONS, key=lambda a: reward(a, e)); mr = []
                if chat and onlinemem:
                    qd = np.argsort([np.linalg.norm(m[0] - svec(e)) for m in onlinemem])[:args.topk]; mr = [onlinemem[k] for k in qd]
                ch = {"oracle": best, "oracle_tuned_test": otr(e), "fixed_rule": fixed_rule(e),
                      "online_rule": online_rule(e, onwin), "gbdt": str(gbdt.predict([svec(e)])[0]),
                      "cart": str(cart.predict([svec(e)])[0]),
                      "bandit": (ACTIONS[int(q.argmax())] if rng.random() > 0.15 else ACTIONS[int(rng.integers(4))])}
                if chat: ch["llm_online_mem"], _ = chat(e, mr, rs)
                onwin.append(e); onlinemem.append((svec(e), e["SNR"], best))
                ba = ACTIONS.index(ch["bandit"]); ncq[ba] += 1; q[ba] += (reward(ch["bandit"], e) - q[ba]) / ncq[ba]
                for m in methods: acts[m].append(ch[m])
            for m in methods:
                A = acts[m]
                sm[m]["overallJ"].append(float(np.mean([reward(A[i], rows[i]) for i in range(len(rows))])))
                sm[m]["regret"].append(float(np.mean([reward(max(ACTIONS, key=lambda a: reward(a, rows[i])), rows[i]) - reward(A[i], rows[i]) for i in range(len(rows))])))
                unk = [i for i in range(len(rows)) if rows[i]["is_unknown"]]; rej = [i for i in range(len(rows)) if A[i] == "A4"]
                sm[m]["unk_reject_recall"].append(float(np.mean([A[i] == "A4" for i in unk])) if unk else 0.0)
                sm[m]["unk_reject_prec"].append(float(np.mean([rows[i]["is_unknown"] for i in rej])) if rej else 0.0)
                lowk = [i for i in range(len(rows)) if (not rows[i]["is_unknown"]) and rows[i]["SNR"] <= 0]
                sm[m]["false_reject_lowSNRknown"].append(float(np.mean([A[i] == "A4" for i in lowk])) if lowk else 0.0)
                sm[m]["switch_rate"].append(float(np.mean([A[i] != A[i - 1] for i in range(1, len(A))])))
        for m in methods:
            for k in METRICS: perseed[m][k].append(float(np.mean(sm[m][k])))
        if chat: print(f"  seed {si}: llm J={perseed['llm_online_mem']['overallJ'][-1]:.3f} | gbdt {perseed['gbdt']['overallJ'][-1]:.3f} fixed {perseed['fixed_rule']['overallJ'][-1]:.3f}", flush=True)

    def ci(v):
        a = np.array(v, float); return float(a.mean()), (float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0)
    rep = {"dataset": args.dataset, "model": model, "n_seeds": args.n_seeds, "train_seed": cfg.seed,
           "unseen_train": args.unseen_train, "unseen_test": args.unseen_test,
           "train_schedules": TRAIN_SCHED, "test_schedules": TEST_SCHED, "trainJ_tuned": {k: float(v) for k, v in trainJ.items()},
           "metrics": {m: {k: {"mean": ci(perseed[m][k])[0], "ci": ci(perseed[m][k])[1]} for k in METRICS} for m in methods}}
    rep["transfer_gap"] = {m: float(trainJ[m] - rep["metrics"][m]["overallJ"]["mean"]) for m in ["fixed_rule", "gbdt", "cart"]}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"transfer_{tag}{args.dataset}.json"), "w"), indent=2)
    print(f"\n=== FROZEN-POLICY TRANSFER ({model}, {args.dataset}, {args.n_seeds} seeds, held-out unknown+drift) ===")
    print("%-18s %-13s %-13s %-13s %-13s %-13s" % ("method", "testJ", "regret", "unkRecall", "unkPrec", "falseRejKnown"))
    for m in methods:
        mm = rep["metrics"][m]
        print("  %-16s %5.3f+-%.3f  %5.3f+-%.3f  %5.3f+-%.3f  %5.3f+-%.3f  %5.3f+-%.3f" % (
            m, mm["overallJ"]["mean"], mm["overallJ"]["ci"], mm["regret"]["mean"], mm["regret"]["ci"],
            mm["unk_reject_recall"]["mean"], mm["unk_reject_recall"]["ci"], mm["unk_reject_prec"]["mean"], mm["unk_reject_prec"]["ci"],
            mm["false_reject_lowSNRknown"]["mean"], mm["false_reject_lowSNRknown"]["ci"]))
    print("transfer_gap (trainJ-testJ, tuned controllers; bigger=worse transfer):", {k: round(v, 3) for k, v in rep["transfer_gap"].items()})
    open(os.path.join(out_dir, f"TRANSFER_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
