#!/bin/bash
# Stage C: recipe ablations on the chosen languages (en, hi, te, ur).
# Reference = stage B recipe (B-ur): akshara-first (min 300), eval x300 + 600K general words, alpha_min 30.
# Each run changes ONE thing and re-balances the weights, so levels are comparable.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 RAYON_NUM_THREADS=2
L=en,hi,te,ur
M='{"en": 1.0647, "hi": 0.6865, "te": 2.0599, "ur": 0.6642}'
run() { python src/search.py --stage "$1" --langs $L --cfg "$2" --eval-w "$3" --ext-w "$4" --iters "$5" --mult "$M" >> "experiments/logs/$1.out" 2>> "experiments/logs/$1.err"; }
jobs_list=(
 'C-plain|{"alpha_min":30,"akshara":false}|300|1|10'
 'C-akfew|{"alpha_min":30,"akshara":true,"akshara_min":3000}|300|1|10'
 'C-ew100|{"alpha_min":30,"akshara":true,"akshara_min":300}|100|1|10'
 'C-ew1000|{"alpha_min":30,"akshara":true,"akshara_min":300}|1000|1|10'
 'C-ext150|{"alpha_min":30,"akshara":true,"akshara_min":300,"ext_words":150000}|300|1|10'
 'C-bytefb|{"alpha_min":30,"akshara":true,"akshara_min":300,"byte_fallback":true}|300|1|10'
 'C-nfkc|{"alpha_min":30,"akshara":true,"akshara_min":300,"norm":"nfkc"}|300|1|10'
 'C-alpha100|{"alpha_min":100,"akshara":true,"akshara_min":300}|300|1|10'
 'C-evalonly|{"akshara":true}|1|0|12'
)
for j in "${jobs_list[@]}"; do
  IFS='|' read -r st cfg ew xw it <<< "$j"
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 2; done
  run "$st" "$cfg" "$ew" "$xw" "$it" &
done
wait
echo ALLDONE
