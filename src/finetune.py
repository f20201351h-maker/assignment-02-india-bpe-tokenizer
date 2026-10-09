"""Final stage: small local search over the language weights to shrink the spread further.

Starting from the balanced weights, each trial perturbs the four multipliers slightly (half
of the trials move in the direction that should close the gap, half are random) and keeps
the result if the spread got smaller. The step size shrinks after repeated failures.
Every trial is a full retrain + evaluation and is written to the ledger.

    python src/finetune.py --stage D-ft1 --seed 1 --trials 30 --cfg '{...}' --mult '{...}'
"""
import argparse
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from search import ROOT, fmt, make_weights, run_one  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
ap.add_argument("--langs", default="en,hi,te,ur")
ap.add_argument("--cfg", required=True)
ap.add_argument("--mult", required=True)
ap.add_argument("--eval-w", type=float, default=300.0)
ap.add_argument("--ext-w", type=float, default=1.0)
ap.add_argument("--trials", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sigma", type=float, default=0.01)
ap.add_argument("--elastic", type=float, default=-0.2, help="assumed d log(fertility) / d log(weight)")
a = ap.parse_args()

langs = a.langs.split(",")
cfg = json.loads(a.cfg)
rng = random.Random(a.seed)
log_f = open(ROOT / "experiments" / "logs" / f"{a.stage}.log", "a", encoding="utf-8")


def log(m):
    print(m, flush=True)
    log_f.write(m + "\n")
    log_f.flush()


def ferts(rec):
    return {l: rec["result"]["eval"][l]["fertility"] for l in langs}


best_m = json.loads(a.mult)
rec, _ = run_one(a.stage, cfg, langs, make_weights(langs, best_m, a.eval_w, a.ext_w), note="start")
best_s, best_f = rec["result"]["spread"], ferts(rec)
log(f"start {fmt(rec)}")
sigma, fails = a.sigma, 0
elastic = a.elastic
for t in range(a.trials):
    mean = sum(math.log(v) for v in best_f.values()) / len(langs)
    m = {}
    for l in langs:
        step = rng.gauss(0, sigma)
        if t % 2 == 0:  # directional proposal: push weight towards closing this language's gap
            step += -(math.log(best_f[l]) - mean) / elastic * rng.uniform(0.2, 1.0)
        m[l] = best_m[l] * math.exp(step)
    g = math.exp(sum(math.log(v) for v in m.values()) / len(m))
    m = {l: v / g for l, v in m.items()}
    rec, _ = run_one(a.stage, cfg, langs, make_weights(langs, m, a.eval_w, a.ext_w), note=f"trial {t} sigma {sigma:.4g}")
    s = rec["result"]["spread"]
    ok = s < best_s
    log(f"{'*' if ok else ' '} {fmt(rec)} sigma={sigma:.4g}")
    if ok:
        best_s, best_m, best_f, fails = s, m, ferts(rec), 0
    else:
        fails += 1
        if fails >= 3:
            sigma, fails = max(sigma * 0.6, 2e-4), 0
log(f"== DONE {a.stage} best_spread={best_s} mult={json.dumps(best_m)}")
