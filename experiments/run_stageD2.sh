#!/bin/bash
# Stage D (restart of the fine-tuning chains with calibrated steps; D-ft1..4 overshot and were stopped)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 RAYON_NUM_THREADS=2
CFG='{"alpha_min":30,"akshara":true,"akshara_min":3000}'
M=$(cat experiments/akfew_mult.json)
for s in 5 6 7 8; do
  python src/finetune.py --stage D-ft$s --seed $s --trials 30 --sigma 0.0015 --elastic -1.0 --cfg "$CFG" --mult "$M" > experiments/logs/D-ft$s.out 2> experiments/logs/D-ft$s.err &
done
wait
echo ALLDONE
