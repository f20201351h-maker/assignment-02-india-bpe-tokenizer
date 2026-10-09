#!/bin/bash
# Stage B: fourth-language comparison. Same recipe for every candidate (akshara-first BPE with aksharas of weighted count >= 300 plus all eval-page aksharas,
# eval pages x300 + 600K words of general Wikipedia text per language, rare chars < 30 dropped),
# then per-language weights balanced. Plus the en/hi/te-only lower bound.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 RAYON_NUM_THREADS=2
CFG='{"alpha_min":30,"akshara":true,"akshara_min":300}'
run() { python src/search.py --stage "$1" --langs "$2" --cfg "$CFG" --eval-w 300 --ext-w 1 --iters 10 --mult "$3" > "experiments/logs/$1.out" 2> "experiments/logs/$1.err"; }
M='{"en":0.94,"hi":0.43,"te":1.84'
jobs_list=(
 "B-3lang|en,hi,te|$M}"
 "B-mr|en,hi,te,mr|$M,\"mr\":1.35}"
 "B-bn|en,hi,te,bn|$M,\"bn\":1.0}"
 "B-ta|en,hi,te,ta|$M,\"ta\":1.0}"
 "B-ml|en,hi,te,ml|$M,\"ml\":1.8}"
 "B-pa|en,hi,te,pa|$M,\"pa\":1.0}"
 "B-or|en,hi,te,or|$M,\"or\":1.5}"
 "B-ur|en,hi,te,ur|$M,\"ur\":1.0}"
 "B-kn|en,hi,te,kn|$M,\"kn\":2.0}"
 "B-gu|en,hi,te,gu|$M,\"gu\":1.5}"
 "B-sa|en,hi,te,sa|$M,\"sa\":1.5}"
)
for j in "${jobs_list[@]}"; do
  IFS='|' read -r st langs mult <<< "$j"
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 2; done
  run "$st" "$langs" "$mult" &
done
wait
echo ALLDONE
