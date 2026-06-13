#!/bin/bash
# Launch full B1b adaptive-oracle diagnostic in detached screens.
CODE=.
RES=../results
LOG=../logs
PY=python
mkdir -p "$RES" "$LOG"
rm -f "$RES"/B1B_DONE_rml2016.flag "$RES"/B1B_DONE_rml2018.flag
rm -f "$RES"/b1b_adaptive_rml2016.json "$RES"/b1b_adaptive_rml2018.json
cd "$CODE" || exit 1
screen -dmS b1b_2016 bash -c "CUDA_VISIBLE_DEVICES=1 $PY scripts/run_b1b_adaptive.py --dataset rml2016 --seeds 42,202,303 2>&1 | tee $LOG/b1b_2016.log"
screen -dmS b1b_2018 bash -c "CUDA_VISIBLE_DEVICES=2 $PY scripts/run_b1b_adaptive.py --dataset rml2018 --seeds 42,202 2>&1 | tee $LOG/b1b_2018.log"
sleep 4
echo "=== screens ==="; screen -ls
echo "=== GPU ==="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
