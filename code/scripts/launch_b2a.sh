#!/bin/bash
# Launch adaptation-oracle headroom gate (RML2016@GPU1, RML2018@GPU2).
CODE=.
RES=../results
LOG=../logs
PY=python
mkdir -p "$RES" "$LOG"
rm -f "$RES"/B2A_DONE_rml2016.flag "$RES"/B2A_DONE_rml2018.flag
rm -f "$RES"/b2a_adaptoracle_rml2016.json "$RES"/b2a_adaptoracle_rml2018.json
cd "$CODE" || exit 1
screen -dmS b2a_2016 bash -c "CUDA_VISIBLE_DEVICES=1 $PY scripts/run_b2a_adaptoracle.py --dataset rml2016 --seeds 42,202,303 2>&1 | tee $LOG/b2a_2016.log"
screen -dmS b2a_2018 bash -c "CUDA_VISIBLE_DEVICES=2 $PY scripts/run_b2a_adaptoracle.py --dataset rml2018 --seeds 42,202 2>&1 | tee $LOG/b2a_2018.log"
sleep 4
echo "=== screens ==="; screen -ls
echo "=== GPU ==="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
