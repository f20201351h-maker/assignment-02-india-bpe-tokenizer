#!/bin/bash
# Stage D: final recipe (C-akfew). (1) four fine-tuning chains, (2) English weight sweep,
# (3) balancing with English pinned at 1.2.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 RAYON_NUM_THREADS=2
CFG='{"alpha_min":30,"akshara":true,"akshara_min":3000}'
M=$(cat experiments/akfew_mult.json)
for s in 1 2 3 4; do
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 2; done
  python src/finetune.py --stage D-ft$s --seed $s --trials 30 --cfg "$CFG" --mult "$M" > experiments/logs/D-ft$s.out 2> experiments/logs/D-ft$s.err &
done
while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 2; done
python - "$CFG" "$M" > experiments/logs/D-entrade.out 2> experiments/logs/D-entrade.err <<'PY' &
import json, sys
sys.path.insert(0, "src")
from search import run_one, make_weights, fmt
cfg, m = json.loads(sys.argv[1]), json.loads(sys.argv[2])
L = ["en", "hi", "te", "ur"]
for f in [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]:
    mm = dict(m); mm["en"] = m["en"] * f
    rec, _ = run_one("D-entrade", cfg, L, make_weights(L, mm, 300, 1), note=f"en weight x{f}")
    print(fmt(rec), flush=True)
PY
while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 2; done
EN12=$(python -c "import json;m=json.load(open('experiments/akfew_mult.json'));m['en']*=4;print(json.dumps(m))")
python src/search.py --stage D-en12 --langs en,hi,te,ur --cfg "$CFG" --eval-w 300 --ext-w 1 --iters 12 --mult "$EN12" --fixed '{"en": 1.2}' --note "English pinned at 1.2" > experiments/logs/D-en12.out 2> experiments/logs/D-en12.err &
wait
echo ALLDONE
