"""Patch run_agent_e2e.py to add 4 new compound scenarios (total 8)."""
fp = "./scripts/run_agent_e2e.py"
with open(fp) as f:
    code = f.read()

old_scn = '    SCENARIOS = ["drift", "novelty", "confusion", "healthy"]'
new_scn = '    SCENARIOS = ["drift", "novelty", "confusion", "healthy",\n                 "severe_drift", "drift_novelty", "subtle_novelty", "noisy_healthy"]'
code = code.replace(old_scn, new_scn)

old_cor = '    CORRECT = {"drift": "adapt", "novelty": "flag", "confusion": "allocate", "healthy": "none"}'
new_cor = ('    CORRECT = {"drift": "adapt", "novelty": "flag", "confusion": "allocate", "healthy": "none",\n'
           '               "severe_drift": "adapt", "drift_novelty": "flag",\n'
           '               "subtle_novelty": "flag", "noisy_healthy": "none"}')
code = code.replace(old_cor, new_cor)

old_batch = '''        def make_batch(scn):
            hi = te[(data.snr[te] >= 6) & np.isin(data.y[te], seen)]
            lo = te[(data.snr[te] <= -4) & np.isin(data.y[te], seen)]
            unk = te[np.isin(data.y[te], list(U))]
            if scn == "healthy":
                idx = rng.permutation(hi)[:args.batch]
            elif scn == "drift":   # low-SNR covariate shift
                idx = rng.permutation(lo)[:args.batch]
            elif scn == "novelty": # inject unknown-class signals
                idx = np.concatenate([rng.permutation(hi)[:args.batch // 2], rng.permutation(unk)[:args.batch // 2]])
            else:                   # confusion: pick a confusable pair, high-SNR
                idx = rng.permutation(hi)[:args.batch]
            return idx'''

new_batch = '''        def make_batch(scn):
            hi = te[(data.snr[te] >= 6) & np.isin(data.y[te], seen)]
            lo = te[(data.snr[te] <= -4) & np.isin(data.y[te], seen)]
            vlo = te[(data.snr[te] <= -8) & np.isin(data.y[te], seen)]
            mid = te[(data.snr[te] >= 0) & (data.snr[te] <= 4) & np.isin(data.y[te], seen)]
            unk = te[np.isin(data.y[te], list(U))]
            if scn == "healthy":
                idx = rng.permutation(hi)[:args.batch]
            elif scn == "drift":
                idx = rng.permutation(lo)[:args.batch]
            elif scn == "novelty":
                idx = np.concatenate([rng.permutation(hi)[:args.batch // 2], rng.permutation(unk)[:args.batch // 2]])
            elif scn == "severe_drift":
                pool = vlo if len(vlo) >= args.batch else lo
                idx = rng.permutation(pool)[:args.batch]
            elif scn == "drift_novelty":
                idx = np.concatenate([rng.permutation(lo)[:args.batch // 2], rng.permutation(unk)[:args.batch // 2]])
            elif scn == "subtle_novelty":
                n_unk = max(1, args.batch // 5)
                idx = np.concatenate([rng.permutation(hi)[:args.batch - n_unk], rng.permutation(unk)[:n_unk]])
            elif scn == "noisy_healthy":
                pool = mid if len(mid) >= args.batch else hi
                idx = rng.permutation(pool)[:args.batch]
            else:
                idx = rng.permutation(hi)[:args.batch]
            return idx'''

code = code.replace(old_batch, new_batch)

with open(fp, "w") as f:
    f.write(code)
print("Patched: 8 scenarios (drift, novelty, confusion, healthy, severe_drift, drift_novelty, subtle_novelty, noisy_healthy)")
