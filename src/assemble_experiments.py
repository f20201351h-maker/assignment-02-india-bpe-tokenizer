"""Collect the experiment summaries shown on the website into experiments/site_experiments.json.

All values are read from the ledgers; the only hand-written parts are the short names and
observations attached to each recipe variant.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import summarize as S  # noqa: E402
from pipeline import eval_text, word_counts  # noqa: E402
from textnorm import count_words  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

ABLATION_NAMES = {
    "C-plain2": ("Plain BPE (no akshara rule)", "lower level, but tokens cut through aksharas"),
    "B-ur": ("Akshara-first, aksharas with count ≥ 300 glued", "stage B recipe"),
    "C-akfew": ("Akshara-first, only aksharas seen on the pages glued", "fewer akshara tokens, more room for words"),
    "C2-ak1000": ("Akshara-first, aksharas with count ≥ 1,000 glued", "between the two"),
    "C-ew100": ("Pages counted ×100 instead of ×300", "better on unseen text, worse on the pages"),
    "C-ew1000": ("Pages counted ×1000", "could not be balanced: Telugu jumps"),
    "C2-akfew-ew200": ("Pages ×200 (with page aksharas only)", ""),
    "C2-akfew-ew500": ("Pages ×500 (with page aksharas only)", ""),
    "C-ext150": ("150K instead of 600K general words per language", "could not be balanced as well"),
    "C2-akfew-ext300": ("300K general words per language (with page aksharas only)", ""),
    "C-evalonly": ("Evaluation pages only, no general text", "could not be balanced; worst on unseen text"),
    "C-bytefb": ("Byte fallback (+256 byte tokens)", "costs vocabulary, no gain on these pages"),
    "C-nfkc": ("NFKC instead of NFC normalisation", "no difference, NFC kept (lossless)"),
    "C-alpha100": ("Drop characters seen < 100 times (instead of 30)", "no real difference"),
}


def main(final_stage):
    recs = S.load_all()
    st = S.by_stage(recs)
    out = {}
    pt = word_counts("en", "eval", "nfc", "ms_punct")
    out["punct_floor_en"] = sum(pt.values()) / count_words(eval_text("en"))
    rows = []
    for r in S.ablation_rows(st):
        if r["stage"] not in ABLATION_NAMES:
            continue
        name, note = ABLATION_NAMES[r["stage"]]
        if r["spread"] > 0.01 and "could not" not in note:
            note = (note + "; " if note else "") + f"best spread only {r['spread']:.3f}"
        rows.append({**{k: r[k] for k in ("stage", "level", "max", "spread", "heldout", "akv_eval", "akv_heldout",
                                          "chars", "stage1", "learned", "best_id")},
                     "name": name, "note": note, "chosen": r["stage"] == final_stage})
    rows.sort(key=lambda r: r["max"])
    for r in rows:
        r["level"] = r["max"]
    out["ablations"] = rows
    # English trade-off
    pts = []
    for r in sorted(st.get("D-entrade2", []), key=lambda r: r["weights"]["en.eval"]):   # sweep from the released weights
        f = {l: S.fert(r, l) for l in r["langs"]}
        others = {l: v for l, v in f.items() if l != "en"}
        wl = max(others, key=others.get)
        pts.append({"en_weight": round(float(r["note"].split("x")[-1]), 3), "en": f["en"], "worst_other": others[wl],
                    "worst_lang": wl, "spread": r["result"]["spread"], "score": r["result"]["score"], "id": r["id"]})
    out["tradeoff"] = {"points": sorted(pts, key=lambda p: p["en_weight"])}
    reach = [p for p in out["tradeoff"]["points"] if p["en"] <= 1.2]
    out["tradeoff"]["first_en_le_12"] = reach[0] if reach else None
    out["n_distinct_tokenizers"] = len({r["tokenizer_sha256"] for r in recs})
    out["n_runs"] = len(recs)
    e12 = st.get("D-en12", [])
    if e12:
        best = next(r for r in reversed(e12) if "best+heldout" in r["note"])   # the balancer's best run
        f = {l: S.fert(best, l) for l in best["langs"]}
        wl = max(f, key=f.get)
        out["tradeoff"]["en12"] = {"id": best["id"], "en": f["en"], "worst": f[wl], "worst_lang": wl,
                                   "spread": best["result"]["spread"], "score": best["result"]["score"],
                                   "fertility": f}
    # trace: the balancing runs of the final stage followed by the fine-tuning trials, in time order
    tr = st.get(final_stage, []) + [r for k, v in st.items() if k.startswith("D-ft") for r in v]
    tr.sort(key=lambda r: (r["time"], r["id"]))
    out["trace"] = [{"step": i + 1, "id": r["id"], "spread": r["result"]["spread"],
                     "max": r["result"]["max_fertility"], "min": r["result"]["min_fertility"]}
                    for i, r in enumerate(tr) if r["result"]["spread"] > 0]
    (ROOT / "experiments" / "site_experiments.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                encoding="utf-8", newline="\n")
    print(f"ablations={len(rows)} tradeoff={len(pts)} en12={'yes' if e12 else 'no'} trace={len(out['trace'])}")


if __name__ == "__main__":
    main(sys.argv[1])
