"""Measure per-call planning latency for all local models on 10 intents."""
import argparse, time, os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

INTENTS = [
    "Check recognition accuracy at the current SNR.",
    "Is the recognizer drifting?",
    "Give me a full health report of the recognizer without changing anything.",
    "Detect drift and recalibrate if needed.",
    "Are there unknown modulations in the stream? If so, flag them.",
    "The system seems unstable. Check accuracy, drift, and confusion. Adapt only if drift is confirmed.",
    "Probe for confused modulation pairs and allocate budget to fix them.",
    "Look for unknowns and drift together before deciding anything.",
    "Recalibrate the classifier to handle SNR drop.",
    "What is the current recognition accuracy at low SNR?",
]

SYS = ("You are an adaptive AMR controller. Tools: check_acc, detect_drift, confusion_probe, "
       "openset_reject, adapt, allocate_budget. Return a JSON list of tool names to run.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model_path", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_warmup", type=int, default=2)
    ap.add_argument("--n_runs", type=int, default=3)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = os.path.basename(args.hf_model_path.rstrip("/"))
    print(f"Loading {model_name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.hf_model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path, torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    print(f"Loaded on {args.device}", flush=True)

    def generate(intent):
        text = f"[system] {SYS}\n[user] {intent}"
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(mdl.device) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=120, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # warmup
    for i in range(args.n_warmup):
        generate(INTENTS[0])
    print(f"Warmup done ({args.n_warmup} calls)", flush=True)

    latencies = []
    for run in range(args.n_runs):
        for intent in INTENTS:
            t0 = time.perf_counter()
            generate(intent)
            dt = time.perf_counter() - t0
            latencies.append(dt)

    lat = sorted(latencies)
    n = len(lat)
    result = {
        "model": model_name,
        "device": args.device,
        "n_intents": len(INTENTS),
        "n_runs": args.n_runs,
        "n_total": n,
        "median_s": lat[n // 2],
        "mean_s": sum(lat) / n,
        "p5_s": lat[int(n * 0.05)],
        "p95_s": lat[int(n * 0.95)],
        "p99_s": lat[int(n * 0.99)],
        "min_s": lat[0],
        "max_s": lat[-1],
    }
    print(f"\n{'='*60}")
    print(f"LATENCY: {model_name}")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:12s} = {v:.3f}")
        else:
            print(f"  {k:12s} = {v}")

    out_dir = os.path.join(ROOT, "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = model_name.lower().replace("-", "").replace("_", "")
    json.dump(result, open(os.path.join(out_dir, f"latency_{tag}.json"), "w"), indent=2)
    print(f"Saved to results/latency_{tag}.json")

if __name__ == "__main__":
    main()
