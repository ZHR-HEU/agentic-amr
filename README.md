# Agentic AMR

**Zero-Shot Compositional Tool Planning for Adaptive Online Modulation Recognition with an IQ-Free LLM Controller**

![status](https://img.shields.io/badge/status-under%20review-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> ⚠️ This repository accompanies a manuscript **currently under review**. The code and results are released for transparency and reproducibility; APIs and numbers may still change.

---

## Overview

This project studies **online automatic modulation recognition (AMR)** under non-stationary
conditions (changing SNR, channel drift, and emerging classes), where labels arrive under a
fixed per-step budget and the recognizer must keep adapting.

The central idea is a clean **separation of roles**:

- A small **CNN recognizer** does all the signal classification on raw IQ.
- A frozen **LLM controller** acts only as a *control plane*: it reads an **IQ-free RF State / Candidate Card** (compact numerical/textual summaries — accuracy, calibration, drift signals, confusion structure, class balance, label budget) and **plans over deterministic RF tools**, emitting soft weights over acquisition criteria.

The LLM **never sees raw IQ samples and never predicts a modulation label.** It orchestrates
diagnostic tools — accuracy checking, drift detection, confusion probing, open-set rejection,
adaptation, and label-budget allocation — while all thresholded numerical actions stay
deterministic. This keeps the LLM's role auditable and avoids handing safety-critical numeric
decisions to a stochastic model.

```
   raw IQ ──► CNN recognizer ──► predictions, logits, calibration
                                        │
                                        ▼
                          IQ-free RF State / Candidate Card
                                        │
                                        ▼
                    LLM controller  (plans tools, emits criterion weights)
                                        │
                                        ▼
        deterministic tools: select ─► query labels ─► update recognizer
```

## Repository Layout

```text
code/
  amrl/              core library
    data.py            RML2016 / RML2018 loading, streaming, subsampling
    model.py           CNN / CNN-GRU recognizers
    state.py           IQ-free RF State / Candidate Card construction
    controllers.py     acquisition controllers (incl. the LLM controller + registry)
    selection.py       criterion scores (entropy, margin, coreset, class balance)
    rf_features.py     RF feature / criterion computation
    rf_systems.py      RF-tool definitions
    regimes.py         drift regimes (snr_ramp, snr_step, channel_drift, ...)
    adapt.py           incremental update / replay
    metrics.py         accuracy, ECE, labels-to-target, etc.
    episode.py         online active-learning episode loop
    spectrum_kb.py     modulation / spectrum knowledge for RAG-style probes
  configs/default.yaml  single source of truth for all hyper-parameters
  scripts/             experiment, ablation, and plotting entry points
results/               JSON result artifacts, summary, and prompt/response traces
figs/                  figures
docs/                  reproducibility notes
```

## Installation

```bash
cd code
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10. Install a CUDA-enabled PyTorch build to train the CNN recognizer on GPU.

## Data

The repository does **not** redistribute the RadioML datasets. Obtain them separately and place
local copies at:

```text
data/RML2016.10a_dict.pkl          # RML2016.10A — 11 classes, length-128
data/GOLD_XYZ_OSC.0001_1024.hdf5   # RML2018.01A — 24 classes, length-1024 (subsampled)
```

or point the config at your own paths:

```bash
python scripts/run_online_al.py --set dataset.rml2016_path=/path/to/RML2016.10a_dict.pkl
```

## LLM Endpoint

The `llm` controller talks to any **OpenAI-compatible** endpoint (e.g. a local vLLM server) via
the `openai` Python client. Defaults (override in `code/configs/default.yaml`):

```yaml
controller:
  endpoint: http://localhost:8000/v1
  model: Qwen3-8B
  temperature: 0.0
  enable_thinking: false       # keep reasoning traces off so JSON parses cleanly
  llm_fallback: fixed_uniform  # used if the LLM call/parse fails
```

For local Hugging Face generation instead of an endpoint, pass `--backend hf --hf_model_path /path/to/model`.
**No endpoint is needed** for the non-LLM controllers (see table below).

## Quick Start

All commands run from `code/`.

```bash
# 1. Sanity-check the pipeline (no LLM required)
python scripts/sanity_check.py --dataset rml2016

# 2. Run one online active-learning episode (deterministic controller)
python scripts/run_online_al.py --set controller.name=fixed_hybrid stream.regime=snr_ramp

# 3. Intent-to-tool benchmark with an OpenAI-compatible LLM endpoint
python scripts/run_agent_intent.py --backend openai --tag qwen3_8b
python scripts/compute_bootstrap_ci.py

# 4. Closed-loop probe (5 seeds)
python scripts/run_agent_e2e.py --dataset rml2016 --n_seeds 5 --backend openai --tag qwen3_8b

# 5. Regenerate the main figures
python scripts/plot_fig_main.py ../figs
```

Scripts write JSON artifacts to `results/` by default.

## Controllers

The acquisition controller decides how the per-step label budget is spent. Available via
`--set controller.name=<name>`:

| Name | Endpoint? | Description |
|------|:---------:|-------------|
| `random` | no | uniform random acquisition |
| `entropy` / `margin` / `coreset` / `class_balance` | no | single-criterion baselines |
| `fixed_uniform` | no | equal blend of all criteria |
| `fixed_hybrid` | no | fixed hand-tuned criterion blend |
| `rule` | no | calibration/drift-gated rule-based controller |
| `llm` | **yes** | LLM emits soft criterion weights from the RF State Card |
| `llm_hardselect` | **yes** | LLM selects criteria directly (hard selection) |

## Main Reported Results

Paper-level values are stored in [`results/paper_main_results.json`](results/paper_main_results.json).

- **Intent benchmark:** five 7B–30B controllers reach novel tool-set F1 from `0.856` to `0.921`.
- **Best trained router baseline:** `0.567` novel F1.
- **Compound-trained router:** `0.649` novel F1.
- **Blind relabeling agreement:** Fleiss' κ `0.894`.
- **Closed-loop correct-action rate:** Qwen3-30B-A3B reaches `0.375` vs. `0.250` for deterministic baselines.

These support the intended boundary: LLMs handle **diagnostic tool orchestration**, while
thresholded numerical actions remain **deterministic**.

## Reproducibility

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). The `results/` directory keeps both the
paper-level summary and raw/intermediate artifacts (including `dynroute_io/` and `fusion_io/`
prompt/response traces) for auditability. Large datasets, checkpoints, and local model weights
are intentionally excluded.

## Citation

This work is a manuscript **currently under review** and has not yet been published. If this
repository is useful, you may cite the code release:

```bibtex
@misc{ji2026agenticamr,
  title  = {Agentic AMR: Zero-Shot Compositional Tool Planning for Adaptive Online
            Modulation Recognition with an IQ-Free LLM Controller},
  author = {Ji, Min and Sun, Lu and Zha, Haoran and Lin, Yun},
  year   = {2026},
  howpublished = {\url{https://github.com/ZHR-HEU/agentic-amr}},
  note   = {Manuscript under review}
}
```

## License

Released under the [MIT License](LICENSE). The RadioML datasets are **not** covered by this
license and remain subject to their original terms.
