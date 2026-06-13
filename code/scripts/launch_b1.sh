#!/bin/bash
# Launch full B1 oracle sweep in detached screens (RML2016@GPU0, RML2018@GPU2).
CODE=.
RES=../results
LOG=../logs
PY=python
mkdir -p "$RES" "$LOG"
# clear stale flags / partial results from earlier validation runs
rm -f "$RES"/B1_DONE_rml2016.flag "$RES"/B1_DONE_rml2018.flag
rm -f "$RES"/b1_oracle_rml2016.json "$RES"/b1_oracle_rml2018.json
cd "$CODE" || exit 1
screen -dmS b1_2016 bash -c "CUDA_VISIBLE_DEVICES=0 $PY scripts/run_b1_oracle.py --dataset rml2016 --seeds 42,202,303 2>&1 | tee $LOG/b1_2016.log"
screen -dmS b1_2018 bash -c "CUDA_VISIBLE_DEVICES=2 $PY scripts/run_b1_oracle.py --dataset rml2018 --seeds 42,202 2>&1 | tee $LOG/b1_2018.log"
sleep 4
echo "=== screens ==="; screen -ls
echo "=== b1_2016.log head ==="; sleep 6; tail -3 "$LOG/b1_2016.log" 2>/dev/null
