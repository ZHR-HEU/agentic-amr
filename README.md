# Agentic AMR

Code and experiment artifacts for **Zero-Shot Compositional Tool Planning for Adaptive Online Modulation Recognition with an IQ-Free LLM Controller**.

The project studies an online automatic modulation recognition (AMR) setting where a CNN performs signal classification and an LLM controller only plans over deterministic RF tools. The LLM never sees raw IQ samples or modulation labels; it receives an IQ-free RF State/Candidate Card and selects tools such as accuracy checking, drift detection, confusion probing, open-set rejection, adaptation, and label-budget allocation.

## Repository Layout

```text
code/
  amrl/              core data, model, state, controller, and RF-tool modules
  configs/           YAML configuration
  scripts/           experiment and plotting scripts
results/             JSON result artifacts and summary
figs/                figures
docs/                reproducibility notes
```

## Installation

```bash
cd code
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build if you plan to train CNN recognizers on GPU.

## Data

The repository does not redistribute RadioML datasets. Place local copies at:

```text
data/RML2016.10a_dict.pkl
data/GOLD_XYZ_OSC.0001_1024.hdf5
```

or edit `code/configs/default.yaml`.

## Quick Start

Run a non-LLM sanity check:

```bash
cd code
python scripts/sanity_check.py --dataset rml2016
```

Run the intent-to-tool benchmark with an OpenAI-compatible local LLM endpoint:

```bash
python scripts/run_agent_intent.py --backend openai --tag qwen3_8b
python scripts/compute_bootstrap_ci.py
```

Run the closed-loop probe:

```bash
python scripts/run_agent_e2e.py --dataset rml2016 --n_seeds 5 --backend openai --tag qwen3_8b
```

Regenerate main figures:

```bash
python scripts/plot_fig_main.py ../figs
```

## Main Reported Results

The paper-level values are stored in `results/paper_main_results.json`.

- Intent benchmark: five 7B--30B controllers reach novel tool-set F1 from `0.856` to `0.921`.
- Best trained router baseline: `0.567` novel F1.
- Compound-trained router: `0.649` novel F1.
- Blind relabeling agreement: Fleiss' kappa `0.894`.
- Closed-loop correct-action rate: Qwen3-30B-A3B reaches `0.375`; deterministic baselines are `0.250`.

These results support the intended boundary: LLMs are used for diagnostic tool orchestration, while thresholded numerical actions remain deterministic.

## Citation

This work is a manuscript currently under review and has not yet been published.
If this repository is useful, you may reference it as:

```bibtex
@unpublished{ji2026agenticamr,
  title={Zero-Shot Compositional Tool Planning for Adaptive Online Modulation Recognition with an IQ-Free LLM Controller},
  author={Ji, Min and Sun, Lu and Zha, Haoran and Lin, Yun},
  year={2026},
  note={Manuscript under review}
}
```

## Notes

- Local model paths and dataset paths in the private experiment environment have been replaced with public placeholders.
- The `results/` directory includes both paper-level summaries and raw/intermediate artifacts for auditability.
- Large datasets, checkpoints, and local model weights are intentionally excluded.
