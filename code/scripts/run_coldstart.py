#!/usr/bin/env python
"""COLD-START label-budget gate (Codex-reviewed design). The scarce resource is
K_target = # TARGET-REGIME oracle-ACTION labels available AFTER deployment to a NEW
regime (held-out unknown families + new drift schedules). Supervised controllers are
trained on PLENTIFUL source-regime labels, then adapted with only K_target target labels.
The zero-shot LLM needs ZERO target labels (K-independent). Honest claim: with enough
K_target the tuned GBDT matches/beats the LLM (DPI ceiling); under cold-start (small
K_target) the label-free interpretable LLM-agent wins, until a crossover K*.

Controllers: oracle | frozen_source_gbdt (source only) | adapted_gbdt(K) (source+K target) |
active_gbdt(K) (source + uncertainty-sampled K target) | no_label_rule (source-tuned) |
rule(K) (tuned on source+K) | zero_shot_llm (no labels) | fewshot_llm(K) (K target
state->action examples in-context). Reports J-vs-K curves + decomposed open-set metrics
+ K* crossover, over multiple seeds. LLM via openai (vLLM).
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
def snr_at(sched, t, S):
    seg, u = t // S, (t % S) / S
    if sched == "abrupt":   return [16, -10, 14, -8][seg % 4]
    if sched == "ramp":     return [16, 16 - 24 * u, -8, -8 + 22 * u][seg]
    if sched == "recurrent":return 4 + 12 * np.sin(2 * np.pi * t / S)
    if sched == "gradcrash":return [14, 14 - 8 * u, 6 - 22 * u, -16 + 30 * u][seg]
    return 0.0
SRC_SCHED = ["abrupt", "ramp"]; TGT_SCHED = ["recurrent", "gradcrash"]


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def reward(a, e):
    l1, l2, l3 = LAM; comp, lat = COST[a]
    if a == "A4": ok = e["is_unknown"]
    elif a == "A1": ok = e["c1"] and not e["is_unknown"]
    elif a == "A2": ok = e["c2"] and not e["is_unknown"]
    else: ok = e["c3"] and not e["is_unknown"]
    return (1.0 if ok else 0.0) - l1 * comp - l2 * lat - l3 * (0.0 if ok else 1.0)


def oracle_act(e): return max(ACTIONS, key=lambda a: reward(a, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rml2018")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seg", type=int, default=35)
    ap.add_argument("--n_test_seeds", type=int, default=5)
    ap.add_argument("--n_label_subsets", type=int, default=15)
    ap.add_argument("--Ks", default="0,16,64,256")
    ap.add_argument("--fewshot_Ks", default="16,64")   # which K to also run few-shot LLM (cost)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--unseen_src", default="")
    ap.add_argument("--unseen_tgt", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config, [f"dataset.name={args.dataset}", "dataset.normalize=per_sample", "dataset.test_per_class=200"])
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    data = load_dataset(cfg); names = data.classes; L = data.length
    if not args.unseen_src: args.unseen_src = "OOK,OQPSK" if args.dataset == "rml2018" else "WBFM,GFSK"
    if not args.unseen_tgt: args.unseen_tgt = "FM,GMSK" if args.dataset == "rml2018" else "PAM4,CPFSK"
    U_s = [names.index(u) for u in args.unseen_src.split(",") if u in names]
    U_t = [names.index(u) for u in args.unseen_tgt.split(",") if u in names]
    seen = [c for c in range(len(names)) if c not in set(U_s) | set(U_t)]
    remap = {c: i for i, c in enumerate(seen)}; inv = {i: c for c, i in remap.items()}
    crop = args.crop if args.crop > 0 else min(128, L)
    print(f"== COLDSTART == {args.dataset} crop={crop}/{L} seen={len(seen)} U_src={sorted(U_s)} U_tgt={sorted(U_t)}", flush=True)

    tr = data.train_idx[np.isin(data.y[data.train_idx], seen)]
    Xtr, ytr = data.X[tr], np.array([remap[c] for c in data.y[tr]])
    a1 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=cnn"]), len(seen))
    a3 = Classifier(load_config(args.config, [f"dataset.name={args.dataset}", "model.backbone=resnet"]), len(seen))
    set_seed(cfg.seed); print("[train A1]", flush=True); a1.fit(Xtr[:, :, :crop], ytr, args.epochs)
    set_seed(cfg.seed); print("[train A3]", flush=True); a3.fit(Xtr, ytr, args.epochs)
    proto = np.stack([a1.features(Xtr[ytr == i][:300][:, :, :crop]).mean(0) for i in range(len(seen))])
    te_pool = data.test_idx

    def make_env(gi):
        x = data.X[gi:gi + 1]; true = int(data.y[gi]); unk = true not in seen
        p1 = a1.predict_proba(x[:, :, :crop])[0]; lg = a1._logits(x[:, :, :crop]).numpy()[0]; f1 = a1.features(x[:, :, :crop])[0]
        p2 = a1.predict_proba(x)[0]; p3 = a3.predict_proba(x)[0]
        return {"true": true, "is_unknown": unk,
                "c1": (inv[int(p1.argmax())] == true) and not unk, "c2": (inv[int(p2.argmax())] == true) and not unk,
                "c3": (inv[int(p3.argmax())] == true) and not unk,
                "C": float(p1.max()), "H": float(-(p1 * np.log(p1 + 1e-12)).sum()),
                "E": float(-(np.log(np.exp(lg - lg.max()).sum()) + lg.max())),
                "D": float(np.linalg.norm(proto - f1, axis=1).min()), "SNR": int(data.snr[gi])}

    def sample_idx(snr_t, unk_pool):
        ids = unk_pool if unk_pool is not None else seen
        cand = te_pool[np.isin(data.y[te_pool], list(ids)) & (np.abs(data.snr[te_pool] - snr_t) <= 2.5)]
        if len(cand) == 0: cand = te_pool[np.isin(data.y[te_pool], list(ids))]
        return int(rng.choice(cand))

    def gen_stream(scheds, unk_pool, seed, reps=1):
        rg = np.random.default_rng(seed); S = args.seg; rows = []
        for _ in range(reps):
            for sd in scheds:
                for t in range(4 * S):
                    snr = float(snr_at(sd, t, S)); unk = (rg.random() < 0.4) and (snr < 6)
                    rows.append(make_env(sample_idx(snr, unk_pool if unk else None)))
        return rows

    calib = [make_env(int(gi)) for c in seen for gi in rng.permutation(te_pool[data.y[te_pool] == c])[:10]]
    calset = {k: np.array([e[k] for e in calib], float) for k in SKEYS}
    def pct(k, v): return float((calset[k] < v).mean())
    def svec(e): return np.array([pct(k, e[k]) for k in SKEYS])

    # SOURCE-regime labeled pool (plentiful) + TARGET adapt pool (rationed by K)
    src_rows = gen_stream(SRC_SCHED, U_s, cfg.seed + 700, reps=3)
    tgt_pool = gen_stream(TGT_SCHED, U_t, cfg.seed + 800, reps=4)   # target-regime oracle-labeled candidates
    Xsrc = np.array([svec(e) for e in src_rows]); ysrc = np.array([oracle_act(e) for e in src_rows])
    Xtp = np.array([svec(e) for e in tgt_pool]); ytp = np.array([oracle_act(e) for e in tgt_pool])

    from sklearn.ensemble import HistGradientBoostingClassifier
    def fit_gbdt(X, y):
        return HistGradientBoostingClassifier(max_iter=200, max_depth=4).fit(X, y) if len(set(y)) > 1 else None
    frozen_src = fit_gbdt(Xsrc, ysrc)                      # source only (K_target=0)
    src_unc = None
    if frozen_src is not None:
        P = frozen_src.predict_proba(Xtp); src_unc = -(P * np.log(P + 1e-12)).sum(1)   # uncertainty for active sampling

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
    no_label_rule = tune_rule(src_rows)   # source-tuned, zero target labels

    # LLM (zero-shot + few-shot via in-context K examples)
    from openai import OpenAI
    cli = OpenAI(base_url=cfg.controller.endpoint, api_key="EMPTY"); model = cfg.controller.model
    SYS = ("You are an EXPLAINABLE cost-aware decision agent for open-set modulation recognition under drift. "
           "A1=light model (confident,in-distribution); A2=extend sampling (LOW SNR); A3=complex model (hard/uncertain "
           "known); A4=reject UNKNOWN (high energy_OOD/prototype_distance). An unknown can look confident: if OOD/distance "
           "high, prefer A4. End 'ANSWER: A1|A2|A3|A4'.")
    def llm_act(e, shots):
        sh = ""
        if shots:
            sh = "\nLabeled examples (state-percentiles -> correct action):\n" + "\n".join(
                f"  conf P{s[0]:.2f} ent P{s[1]:.2f} energy P{s[2]:.2f} dist P{s[3]:.2f} snr {s[5]:.0f} -> {s[6]}" for s in shots)
        u = (f"State: confidence=P{pct('C',e['C']):.2f}, entropy=P{pct('H',e['H']):.2f}, energy_OOD=P{pct('E',e['E']):.2f}, "
             f"prototype_distance=P{pct('D',e['D']):.2f}, SNR={e['SNR']}dB.{sh}\nChoose. ANSWER: A1|A2|A3|A4.")
        try:
            r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                temperature=0.0, max_tokens=120, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            t = r.choices[0].message.content or ""
        except Exception: return "A1"
        m = re.search(r"ANSWER:\s*(A[1-4])", t) or re.search(r"\b(A[1-4])\b", t); return m.group(1) if m else "A1"

    Ks = [int(x) for x in args.Ks.split(",")]; fsKs = set(int(x) for x in args.fewshot_Ks.split(",") if x)
    test_streams = [gen_stream(TGT_SCHED, U_t, cfg.seed + 1000 + s * 17) for s in range(args.n_test_seeds)]

    def metrics(acts, rows):
        J = float(np.mean([reward(acts[i], rows[i]) for i in range(len(rows))]))
        unk = [i for i in range(len(rows)) if rows[i]["is_unknown"]]; rej = [i for i in range(len(rows)) if acts[i] == "A4"]
        lowk = [i for i in range(len(rows)) if (not rows[i]["is_unknown"]) and rows[i]["SNR"] <= 0]
        return {"J": J, "unkRec": float(np.mean([acts[i] == "A4" for i in unk])) if unk else 0.0,
                "unkPrec": float(np.mean([rows[i]["is_unknown"] for i in rej])) if rej else 0.0,
                "falseRejKnown": float(np.mean([acts[i] == "A4" for i in lowk])) if lowk else 0.0}

    # zero-shot LLM (K-independent): eval once per test seed
    llm_zs = []
    for rows in test_streams:
        acts = [llm_act(e, None) for e in rows]; llm_zs.append(metrics(acts, rows)["J"])
    print(f"[zero_shot_llm] J={np.mean(llm_zs):.3f}+-{np.std(llm_zs):.3f}", flush=True)
    no_label_rule_J = [metrics([no_label_rule(e) for e in rows], rows)["J"] for rows in test_streams]
    frozen_src_J = [metrics([str(frozen_src.predict([svec(e)])[0]) for e in rows], rows)["J"] for rows in test_streams] if frozen_src else [0]*args.n_test_seeds

    curves = {"zero_shot_llm": {"J_mean": float(np.mean(llm_zs)), "J_ci": float(1.96*np.std(llm_zs,ddof=1)/np.sqrt(len(llm_zs)))},
              "no_label_rule": {"J_mean": float(np.mean(no_label_rule_J)), "J_ci": float(1.96*np.std(no_label_rule_J,ddof=1)/np.sqrt(len(no_label_rule_J)))},
              "frozen_source_gbdt": {"J_mean": float(np.mean(frozen_src_J)), "J_ci": float(1.96*np.std(frozen_src_J,ddof=1)/np.sqrt(len(frozen_src_J)))}}
    perK = {}
    for K in Ks:
        agbdt_J, active_J, rule_J, fs_J = [], [], [], []
        for ls in range(args.n_label_subsets):
            rg = np.random.default_rng(cfg.seed + 9000 + K * 7 + ls)
            idxK = rg.permutation(len(tgt_pool))[:K] if K > 0 else np.array([], int)
            # adapted gbdt = source + K target (random)
            if K > 0:
                Xc = np.concatenate([Xsrc, Xtp[idxK]]); yc = np.concatenate([ysrc, ytp[idxK]])
            else: Xc, yc = Xsrc, ysrc
            ag = fit_gbdt(Xc, yc)
            # active gbdt = source + K most-uncertain target
            if K > 0 and src_unc is not None:
                aidx = np.argsort(-src_unc)[:K]; Xa = np.concatenate([Xsrc, Xtp[aidx]]); ya = np.concatenate([ysrc, ytp[aidx]]); acg = fit_gbdt(Xa, ya)
            else: acg = ag
            rl = tune_rule(src_rows + [tgt_pool[i] for i in idxK]) if K > 0 else no_label_rule
            for rows in test_streams[:max(1, args.n_test_seeds // 3) if ls > 2 else args.n_test_seeds]:
                agbdt_J.append(metrics([str(ag.predict([svec(e)])[0]) for e in rows], rows)["J"]) if ag else None
                active_J.append(metrics([str(acg.predict([svec(e)])[0]) for e in rows], rows)["J"]) if acg else None
                rule_J.append(metrics([rl(e) for e in rows], rows)["J"])
        # few-shot LLM at selected K (cost-limited): use K target examples as in-context shots
        if K in fsKs and K > 0:
            for s, rows in enumerate(test_streams[:3]):
                rg = np.random.default_rng(cfg.seed + 5000 + K + s)
                shots_idx = rg.permutation(len(tgt_pool))[:min(K, args.topk)]
                # shot = [C_p, H_p, E_p, D_p, SNR_p, raw_SNR, action]; llm_act uses s[0..3], s[5], s[6]
                shots = [list(Xtp[i]) + [float(tgt_pool[i]["SNR"]), ytp[i]] for i in shots_idx]
                fs_J.append(metrics([llm_act(e, shots) for e in rows], rows)["J"])
        def mc(v): a=np.array(v,float); return (float(a.mean()), float(1.96*a.std(ddof=1)/np.sqrt(len(a)))) if len(a)>1 else (float(a.mean()) if len(a) else 0.0, 0.0)
        m1,c1=mc(agbdt_J); m2,c2=mc(active_J); m3,c3=mc(rule_J)
        perK[K] = {"adapted_gbdt": {"J_mean": m1, "J_ci": c1}, "active_gbdt": {"J_mean": m2, "J_ci": c2},
                   "rule": {"J_mean": m3, "J_ci": c3}}
        if fs_J: m4,c4=mc(fs_J); perK[K]["fewshot_llm"]={"J_mean":m4,"J_ci":c4}
        print(f"[K={K}] adapted_gbdt={m1:.3f} active_gbdt={m2:.3f} rule={m3:.3f}" + (f" fewshot_llm={perK[K].get('fewshot_llm',{}).get('J_mean','-')}" if fs_J else ""), flush=True)

    # K* crossover: smallest K where adapted_gbdt mean >= zero_shot_llm mean
    llm_mean = curves["zero_shot_llm"]["J_mean"]
    kstar = next((K for K in Ks if perK[K]["adapted_gbdt"]["J_mean"] >= llm_mean), None)
    rep = {"dataset": args.dataset, "model": model, "Ks": Ks, "n_test_seeds": args.n_test_seeds,
           "unseen_src": args.unseen_src, "unseen_tgt": args.unseen_tgt,
           "K_independent": curves, "perK": {str(k): v for k, v in perK.items()}, "Kstar_adapted_gbdt": kstar,
           "oracle_J": float(np.mean([metrics([oracle_act(e) for e in rows], rows)["J"] for rows in test_streams]))}
    out_dir = os.path.join(ROOT, cfg.eval.out_dir); os.makedirs(out_dir, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    json.dump(rep, open(os.path.join(out_dir, f"coldstart_{tag}{args.dataset}.json"), "w"), indent=2)
    print(f"\n=== COLD-START label-budget ({model}, {args.dataset}) ===")
    print(f"  oracle={rep['oracle_J']:.3f}  zero_shot_llm={llm_mean:.3f}+-{curves['zero_shot_llm']['J_ci']:.3f}  "
          f"frozen_source_gbdt={curves['frozen_source_gbdt']['J_mean']:.3f}  no_label_rule={curves['no_label_rule']['J_mean']:.3f}")
    for K in Ks:
        p = perK[K]; print(f"  K_target={K:4d}: adapted_gbdt={p['adapted_gbdt']['J_mean']:.3f}+-{p['adapted_gbdt']['J_ci']:.3f} "
                           f"active_gbdt={p['active_gbdt']['J_mean']:.3f} rule={p['rule']['J_mean']:.3f}" + (f" fewshot_llm={p['fewshot_llm']['J_mean']:.3f}" if 'fewshot_llm' in p else ""))
    print(f"  => K* (target labels for adapted_gbdt to catch zero-shot LLM) = {kstar}")
    open(os.path.join(out_dir, f"COLDSTART_DONE_{tag}{args.dataset}.flag"), "w").write("done")


if __name__ == "__main__":
    main()
