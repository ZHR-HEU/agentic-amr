# Reproducibility Notes

## Data

The code expects local copies of:

- RML2016.10A: `data/RML2016.10a_dict.pkl`
- RML2018.01A: `data/GOLD_XYZ_OSC.0001_1024.hdf5`

These files are not redistributed in this repository. Edit `code/configs/default.yaml` or pass overrides with `--set`.

## LLM endpoint

OpenAI-compatible local endpoints are supported through the `openai` Python client. The default configuration assumes:

```yaml
controller:
  endpoint: http://localhost:8000/v1
  model: Qwen3-8B
  temperature: 0.0
  enable_thinking: false
```

For Hugging Face local generation, pass `--backend hf --hf_model_path /path/to/model`.

## Main commands

Run from `code/`:

```bash
pip install -r requirements.txt
python scripts/sanity_check.py --dataset rml2016
python scripts/run_agent_intent.py --backend openai --tag qwen3_8b
python scripts/compute_bootstrap_ci.py
python scripts/run_agent_e2e.py --dataset rml2016 --n_seeds 5 --backend openai --tag qwen3_8b
python scripts/plot_fig_main.py ../figs
```

The scripts save JSON artifacts to `results/` by default.
