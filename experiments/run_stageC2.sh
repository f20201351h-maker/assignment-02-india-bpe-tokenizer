#!/bin/bash
# Stage C2: refinement around the best stage C recipe (C-akfew).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 RAYON_NUM_THREADS=2
L=en,hi,te,ur
M='{"en": 1.0574, "hi": 0.6901, "te": 2.0426, "ur": 0.6709}'
run() { python src/search.py --stage "$1" --langs $L --cfg "$2" --eval-w "$3" --ext-w 1 --iters 10 --mult "$M" >> "experiments/logs/$1.out" 2>> "experiments/logs/$1.err"; }
jobs_list=(
 'C2-ak1000|{"alpha_min":30,"akshara":true,"akshara_min":1000}|300'
 'C2-ak10k|{"alpha_min":30,"akshara":true,"akshara_min":10000}|300'
 'C2-akevalonly|{"alpha_min":30,"akshara":true,"akshara_min":1e15}|300'
 'C2-akfew-ew200|{"alpha_min":30,"akshara":true,"akshara_min":3000}|200'
 'C2-akfew-ew500|{"alpha_min":30,"akshara":true,"akshara_min":3000}|500'
 'C2-akfew-ext300|{"alpha_min":30,"akshara":true,"akshara_min":3000,"ext_words":300000}|300'
)
for j in "${jobs_list[@]}"; do
  IFS='|' read -r st cfg ew <<< "$j"
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 2; done
  run "$st" "$cfg" "$ew" &
done
wait
echo ALLDONE
