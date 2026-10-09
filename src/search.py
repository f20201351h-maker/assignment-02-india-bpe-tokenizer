"""Experiment runner: every trained tokenizer is evaluated and appended to experiments/results.jsonl.

The main tool is `balance()`: for a fixed recipe (data sources, preprocessing, BPE options)
it adjusts one sampling multiplier per language until the four fertilities meet.
Languages whose fertility is above the current mean get more weight, those below get less.
"""
import datetime as dt
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pipeline import DEFAULTS, evaluate, eval_text, tokenizer_sha, train  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "experiments" / "ledger"
LEDGER = ROOT / "experiments" / "results.jsonl"   # stage A probes (run before the per-stage ledgers existed)


def _ledger_for(stage):
    return LEDGER if stage.startswith("A") else LEDGER_DIR / f"{stage}.jsonl"


def _next_id(stage):
    path = _ledger_for(stage)
    if not path.exists():
        return 1
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) + 1


def roundtrip_ok(tok, langs):
    for l in langs:
        t = eval_text(l)
        if tok.decode(tok.encode(t).ids) != tok.normalizer.normalize_str(t):
            return False
    return True


def run_one(stage, cfg, langs, weights, heldout=False, note="", save_to=None):
    t0 = time.time()
    tok, info = train(cfg, langs, weights)
    res = evaluate(tok, langs, heldout=heldout, akshara_check=True)
    rec = {
        "id": (f"E{_next_id(stage):04d}" if stage.startswith("A") else f"{stage}-{_next_id(stage):03d}"),
        "stage": stage, "time": dt.datetime.now().isoformat(timespec="seconds"),
        "langs": langs, "config": {**DEFAULTS, **cfg},
        "weights": {f"{l}.{s}": w for (l, s), w in weights.items()},   # full precision (needed to retrain exactly)
        "info": info, "result": res,
        "checks": {
            "vocab_is_10000": tok.get_vocab_size() == 10_000,
            "no_unk_on_eval": all(res["eval"][l]["unk"] == 0 for l in langs),
            "roundtrip_eval": roundtrip_ok(tok, langs),
        },
        "tokenizer_sha256": tokenizer_sha(tok), "seconds": round(time.time() - t0, 1), "note": note,
    }
    path = _ledger_for(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if save_to:
        pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(save_to))
    return rec, tok


def fmt(rec):
    r = rec["result"]
    f = " ".join(f"{l}={r['eval'][l]['fertility']:.4f}" for l in rec["langs"])
    akv = sum(r["eval"][l].get("akshara_violations", 0) for l in rec["langs"])
    h = ""
    if r.get("heldout"):
        h = " | heldout " + " ".join(f"{l}={r['heldout'][l]['fertility']:.3f}" for l in rec["langs"])
    return (f"{rec['id']} {f} spread={r['spread']:.5f} score={r['score']:.1f} akv={akv} "
            f"chk={'ok' if all(rec['checks'].values()) else rec['checks']}{h}")


def make_weights(langs, mult, eval_w, ext_w):
    w = {}
    for l in langs:
        if eval_w > 0:
            w[(l, "eval")] = eval_w * mult[l]
        if ext_w > 0:
            w[(l, "train")] = ext_w * mult[l]
    return w


def balance(stage, cfg, langs, eval_w=1.0, ext_w=0.0, iters=12, mult=None, note="", log=print,
            final_heldout=True, fixed=None):
    """Iteratively adjust per-language multipliers to equalise fertility. Returns best record.

    fixed: optional {lang: target_fertility}; those languages are steered to the given value
    and the remaining languages are balanced among themselves.
    """
    fixed = fixed or {}
    free = [l for l in langs if l not in fixed]

    def objective(r):
        f = {l: r["eval"][l]["fertility"] for l in langs}
        dev = max(abs(f[l] - t) for l, t in fixed.items()) if fixed else 0.0
        fr = [f[l] for l in free]
        return max(fr) - min(fr) + dev
    mult = dict(mult or {l: 1.0 for l in langs})
    elastic = {l: -0.15 for l in langs}   # initial guess of d log(fertility) / d log(weight)
    prev = None
    best = None
    gain = 0.8                            # step size; halved whenever a step makes things worse
    for it in range(iters):
        rec, _ = run_one(stage, cfg, langs, make_weights(langs, mult, eval_w, ext_w), note=f"{note} it{it}")
        r = rec["result"]
        f = {l: r["eval"][l]["fertility"] for l in langs}
        log(f"  {fmt(rec)}  mult={ {l: round(m, 3) for l, m in mult.items()} }")
        improved = best is None or objective(r) < objective(best[0]["result"])
        if improved:
            best = (rec, dict(mult), f)
        if prev is not None:
            for l in langs:
                dlm = math.log(mult[l]) - math.log(prev[1][l])
                dlf = math.log(f[l]) - math.log(prev[0][l])
                if abs(dlm) > 1e-3:
                    e = dlf / dlm
                    if -2.0 < e < -0.005:
                        elastic[l] = 0.5 * elastic[l] + 0.5 * e
        prev = (f, dict(mult))
        if objective(r) < 1e-4:
            break
        # step from the best point found so far; shrink the step after a failed step
        gain = min(0.8, gain * 1.25) if improved else gain * 0.5
        base_f, mult = best[2], dict(best[1])
        logt = sum(math.log(base_f[l]) for l in free) / len(free)
        for l in langs:
            target = math.log(fixed[l]) if l in fixed else logt
            step = (math.log(base_f[l]) - target) / elastic[l]   # elastic < 0 -> above-target languages get more weight
            mult[l] *= math.exp(max(-1.0, min(1.0, -step * gain)))
        # keep multipliers anchored (the absolute scale of weights does not matter)
        g = math.exp(sum(math.log(m) for m in mult.values()) / len(mult))
        mult = {l: m / g for l, m in mult.items()}
    rec, m = best[0], best[1]
    if final_heldout:
        rec, _ = run_one(stage, cfg, langs, make_weights(langs, m, eval_w, ext_w), heldout=True,
                         note=f"{note} best+heldout")
        log(f"  BEST {fmt(rec)}")
    return rec, m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--langs", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--eval-w", type=float, default=300.0)
    ap.add_argument("--ext-w", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--mult", default=None, help="json dict of starting multipliers")
    ap.add_argument("--note", default="")
    ap.add_argument("--fixed", default=None, help='json dict, e.g. {"en": 1.2}')
    a = ap.parse_args()
    langs = a.langs.split(",")
    out = open(ROOT / "experiments" / "logs" / f"{a.stage}.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        out.write(msg + "\n")
        out.flush()

    log(f"== {a.stage} langs={langs} cfg={a.cfg} eval_w={a.eval_w} ext_w={a.ext_w}")
    rec, m = balance(a.stage, json.loads(a.cfg), langs, eval_w=a.eval_w, ext_w=a.ext_w, iters=a.iters,
                     mult=json.loads(a.mult) if a.mult else None, note=a.note, log=log,
                     fixed=json.loads(a.fixed) if a.fixed else None)
    log(f"== DONE {a.stage} mult={json.dumps(m)}")
