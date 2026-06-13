# Results

This directory contains experiment artifacts for the paper.

- `paper_main_results.json` contains the paper-level values reported in the main tables and figures.
- `transfer_*.json`, `coldstart_*.json`, `interp_*.json`, `openworld_*.json`, and `dynroute_*.json` are raw or intermediate experiment artifacts retained for auditability.
- `dynroute_io/` and `fusion_io/` store prompt/response traces for selected controller probes.

The original RML2016.10A and RML2018.01A datasets are not redistributed. Re-run scripts after placing datasets locally and updating `code/configs/default.yaml`.
