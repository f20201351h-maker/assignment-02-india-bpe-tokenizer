#!/bin/bash
# Stage B (continued after a restart for speed): remaining fourth-language candidates, plus a
# final held-out evaluation at the best weights of the four jobs that had already converged.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 RAYON_NUM_THREADS=2
CFG='{"alpha_min":30,"akshara":true,"akshara_min":300}'
run() { python src/search.py --stage "$1" --langs "$2" --cfg "$CFG" --eval-w 300 --ext-w 1 --iters "$4" --mult "$3" >> "experiments/logs/$1.out" 2>> "experiments/logs/$1.err"; }
M='{"en":0.85,"hi":0.47,"te":2.1'
jobs_list=(
 "B-ml|en,hi,te,ml|$M,\"ml\":2.0}|10"
 "B-pa|en,hi,te,pa|$M,\"pa\":1.0}|10"
 "B-or|en,hi,te,or|$M,\"or\":1.5}|10"
 "B-ur|en,hi,te,ur|$M,\"ur\":0.9}|10"
 "B-kn|en,hi,te,kn|$M,\"kn\":2.0}|10"
 "B-gu|en,hi,te,gu|$M,\"gu\":1.5}|10"
 "B-sa|en,hi,te,sa|$M,\"sa\":1.8}|10"
 "B-3lang|en,hi,te|{\"en\": 0.9707, \"hi\": 0.6621, \"te\": 1.5558}|1"
 "B-bn|en,hi,te,bn|{\"en\": 0.8068, \"hi\": 0.464, \"te\": 2.1245, \"bn\": 1.2574}|1"
 "B-mr|en,hi,te,mr|{\"en\": 1.0152, \"hi\": 0.3776, \"te\": 2.0511, \"mr\": 1.2719}|4"
 "B-ta|en,hi,te,ta|{\"en\": 0.8418, \"hi\": 0.4943, \"te\": 2.1928, \"ta\": 1.0961}|4"
)
for j in "${jobs_list[@]}"; do
  IFS='|' read -r st langs mult it <<< "$j"
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 2; done
  run "$st" "$langs" "$mult" "$it" &
done
wait
echo ALLDONE
