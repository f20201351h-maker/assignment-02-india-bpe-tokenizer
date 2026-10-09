"""Build site/data/site_data.json, site/tokenizer.json and site/tokens.tsv.

Every headline number on the website comes from verify/out/ (the independent re-computation
from the exported tokenizer.json), never from the training harness.
"""
import hashlib
import json
import platform
import re
import shutil
import statistics
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import site_text as T  # noqa: E402
import summarize as S  # noqa: E402

TOK = ROOT / "tokenizer" / "tokenizer.json"
SITE = ROOT / "site"
VOUT = ROOT / "verify" / "out"

LANG_META = {
    "en": {"name": "English", "native": "English", "dir": "ltr", "bcp47": "en"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "dir": "ltr", "bcp47": "hi"},
    "te": {"name": "Telugu", "native": "తెలుగు", "dir": "ltr", "bcp47": "te"},
    "ur": {"name": "Urdu", "native": "اردو", "dir": "rtl", "bcp47": "ur"},
    "mr": {"name": "Marathi", "native": "मराठी", "dir": "ltr", "bcp47": "mr"},
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "dir": "ltr", "bcp47": "pa"},
}


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def ledger(stage):
    p = ROOT / "experiments" / "ledger" / f"{stage}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def first_sentence(text, max_words=18):
    line = text.split("\n")[0]
    words = line.split()
    for i, w in enumerate(words[:max_words]):
        if re.search(r"[.।۔?!]$", w) and i >= 5:
            return " ".join(words[: i + 1])
    return " ".join(words[:max_words])


def fmt_score(x):
    return f"{x:,.1f}"


def bust_caches():
    """Append ?v=<content hash> to the CSS/JS references so browsers never use stale files."""
    def h(rel):
        return hashlib.sha256((SITE / rel).read_bytes()).hexdigest()[:10]
    app = (SITE / "js" / "app.js").read_text(encoding="utf-8")
    app = re.sub(r'from "\./bpe\.js(\?v=[0-9a-f]+)?"', f'from "./bpe.js?v={h("js/bpe.js")}"', app)
    (SITE / "js" / "app.js").write_text(app, encoding="utf-8", newline="\n")
    html = (SITE / "index.html").read_text(encoding="utf-8")
    html = re.sub(r'css/style\.css(\?v=[0-9a-f]+)?"', f'css/style.css?v={h("css/style.css")}"', html)
    html = re.sub(r'js/app\.js(\?v=[0-9a-f]+)?"', f'js/app.js?v={h("js/app.js")}"', html)
    (SITE / "index.html").write_text(html, encoding="utf-8", newline="\n")


def main():
    final = j(ROOT / "tokenizer" / "final_config.json")
    X = j(ROOT / "experiments" / "site_experiments.json")
    langs = final["langs"]
    summ, py, js = j(VOUT / "verification_summary.json"), j(VOUT / "python_report.json"), j(VOUT / "js_report.json")
    if not summ["all_pass"]:
        raise SystemExit("verification did not pass - refusing to build the site")
    sens = j(VOUT / "sensitivity.json")
    manifest = j(ROOT / "data" / "eval" / "manifest.json")
    tok = Tokenizer.from_file(str(TOK))
    tj = j(TOK)
    raw = TOK.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == summ["tokenizer_sha256"]

    # copy artifact + token list
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TOK, SITE / "tokenizer.json")
    # publish the exact evaluation texts (and their manifest) so the score can be re-checked from the website
    (SITE / "data" / "eval").mkdir(parents=True, exist_ok=True)
    for old in (SITE / "data" / "eval").glob("*"):
        old.unlink()
    for l in langs:
        shutil.copyfile(ROOT / "data" / "eval" / f"{l}.txt", SITE / "data" / "eval" / f"{l}.txt")
    shutil.copyfile(ROOT / "data" / "eval" / "manifest.json", SITE / "data" / "eval" / "manifest.json")
    inv = sorted(((i, t) for t, i in tj["model"]["vocab"].items()))
    merges = {a + b: (a, b) for a, b in (tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in tj["model"]["merges"])}
    with open(SITE / "tokens.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("id\ttoken_json\tbuilt_from_json\n")
        for i, t in inv:
            f.write(f"{i}\t{json.dumps(t, ensure_ascii=False)}\t{json.dumps(list(merges[t]), ensure_ascii=False) if t in merges else ''}\n")

    comp = final["composition"]
    held = final.get("heldout", {})
    per = {}
    for l in langs:
        m = manifest["pages"][l]
        text = (ROOT / "data" / "eval" / f"{l}.txt").read_text(encoding="utf-8").rstrip("\n")
        ex = first_sentence(text)
        per[l] = {
            "title": m["title"], "project": m["project"], "revid": m["revid"], "rev_timestamp": m["revision_timestamp"],
            "retrieved_utc": m["retrieved_utc"], "permanent_url": m["permanent_url"], "url": m["url"],
            "clean_text_sha256": m["clean_text_sha256"], "file_sha256": m["file_sha256"],
            "eval_file": f"data/eval/{l}.txt",
            "words": summ["words"][l], "tokens": summ["tokens"][l], "fertility": summ["fertility"][l],
            "fertility_exact": summ["fertility_exact"][l],
            "heldout_fertility": held.get(l, {}).get("fertility"), "heldout_words": held.get(l, {}).get("words"),
            "akshara_violations": 0 if all(c["pass"] for c in summ["checks"] if c["check"].startswith(f"{l}: no token boundary")) else None,
            "example": {"text": ex, "tokens": tok.encode(ex).tokens},
        }
    F = {
        "vocab_size": py["vocab_size_hf"], "composition": comp, "per_lang": per,
        "min_lang": summ["min_lang"], "max_lang": summ["max_lang"],
        "min_fertility": summ["fertility"][summ["min_lang"]], "max_fertility": summ["fertility"][summ["max_lang"]],
        "spread": summ["spread"], "spread_exact": summ["spread_exact"], "score": summ["score"],
        "score_exact": summ["score_exact"], "score_decimal": summ["score_decimal"],
        "tokenizer_sha256": sha, "tokenizer_bytes": len(raw),
        "verification": {
            "agree": summ["all_pass"],
            "summary": (f"{sum(c['pass'] for c in summ['checks'])}/{len(summ['checks'])} checks pass: the Hugging Face "
                        f"library (Python) and a separate JavaScript BPE both reload tokenizer.json and get identical "
                        f"token counts, word counts, spread and score."),
            "checks": summ["checks"], "versions": summ["versions"],
        },
    }
    names = [LANG_META[l]["name"] for l in langs]
    mx, mn = per[F["max_lang"]], per[F["min_lang"]]
    rec = {
        "title_langs": ", ".join(names[:-1]) + " and " + names[-1],
        "langs": langs, "lang_meta": {l: LANG_META[l] for l in langs}, "final": F,
    }
    rec["lede_html"] = (
        f"One BPE vocabulary of exactly 10,000 tokens shared by {rec['title_langs']}. On the four India pages the "
        f"fertilities (tokens per word) range from {mn['fertility']:.6f} ({LANG_META[F['min_lang']]['name']}) to "
        f"{mx['fertility']:.6f} ({LANG_META[F['max_lang']]['name']}). The gap is {F['spread']:.9f} (exactly "
        f"{F['spread_exact']}), so the score is 1000 / gap = {fmt_score(F['score'])}.")
    en = per["en"]["fertility"]

    # ------------------------------------------------------------ experiments
    recs = S.load_all()
    stages = S.by_stage(recs)
    fourth = sorted(S.fourth_rows(stages), key=lambda r: r["max"])
    for r in fourth:
        r["chosen"] = r["lang"] == langs[3]
        r["level"] = r["max"]
        r["note"] = ("chosen" if r["chosen"] else
                     "lower bound: only the three required languages" if r["lang"] == "-" else
                     ("" if r["eligible"] else f"page under {S.MIN_PAGE_WORDS:,} words, shown for comparison"))
    lv = {r["lang"]: r["max"] for r in fourth}
    base = next(r for r in stages["D-base"] if "four pages only" in r["note"])
    base_ext = next(r for r in stages["D-base"] if "general Wikipedia text only" in r["note"])
    trade = X["tradeoff"]
    abl = X["ablations"]
    plain = next(a for a in abl if a["stage"] == "C-plain2")
    chosen_abl = next(a for a in abl if a.get("chosen"))
    w_en = final["weights"]["en.eval"]
    sv = {v["key"]: v for v in sens["variants"]}
    if "current" in sv:
        cur = sv["current"]
        changed = [LANG_META[l]["name"] for l, m in cur["revisions"].items() if not m["same_as_pinned"]]
        when = sens["generated_utc"][:10]
        if abs(cur["spread"] - sv["pinned"]["spread"]) < 1e-12:
            live_sentence = (f"Re-measured on the live pages on {when} ({', '.join(changed) or 'no page'} edited since "
                             f"pinning) the score is unchanged.")
        else:
            live_sentence = (f"Re-measured on the live pages on {when} ({', '.join(changed) or 'no page'} edited since "
                             f"pinning) the score is {cur['score']:,.0f}.")
    else:
        live_sentence = ""
    alt_score = min(sv[k]["score"] for k in ("html_prose", "html_all"))
    alt_delta = max(abs(sv[k]["per_lang"][l] - sv["pinned"]["per_lang"][l]) for k in ("html_prose", "html_all")
                    for l in langs)
    reach = trade["first_en_le_12"]
    held_f = {l: final["heldout"][l]["fertility"] for l in langs}
    lo_h, hi_h = min(held_f, key=held_f.get), max(held_f, key=held_f.get)
    en_margin_tokens = (F["per_lang"]["en"]["fertility"] - F["min_fertility"]) * F["per_lang"]["en"]["words"]
    rec["caveat_html"] = T.CAVEAT.format(
        en=en, reach_en=reach["en"], reach_worst=reach["worst_other"], reach_score=reach["score"], lvl3=lv["-"],
        heldout_range=(f"{held_f[lo_h]:.2f} ({LANG_META[lo_h]['name']}) to {held_f[hi_h]:.2f} "
                       f"({LANG_META[hi_h]['name']})"),
        live_sentence=live_sentence, wiki_score=sv["current_wiki"]["score"],
        alt_lo=min(sv[k]["score"] for k in ("html_prose", "html_all")),
        alt_hi=max(sv[k]["score"] for k in ("html_prose", "html_all")),
        en_margin=en_margin_tokens)
    bytefb = next(a for a in abl if a["stage"] == "C-bytefb")
    bref = next(a for a in abl if a["stage"] == "B-ur")       # same recipe as C-bytefb, without byte fallback
    pw = sorted(((final["weights"][f"{l}.eval"], l) for l in langs), reverse=True)
    rec["method_html"] = T.METHOD.format(
        vocab=f"{py['vocab_size_hf']:,}", n_merges=py["n_merges"], punct_floor=X["punct_floor_en"],
        n_chars=comp["characters"], heldout_unk=sum(v["unk"] for v in final["heldout"].values()),
        bytefb_cost=bytefb["max"] - bref["max"], plain_akv=plain["akv_eval"],
        base_akv=S.akv(base), akshara_cost=chosen_abl["max"] - plain["max"],
        heldout_words=sum(v["words"] for v in final["heldout"].values()),
        heldout_akv=sum(v["akshara_violations"] for v in final["heldout"].values()),
        plain_heldout_akv=plain["akv_heldout"],
        eval_w=final["eval_w"], ext_words=final["config"]["ext_words"],
        page_weights=(", ".join(f"{w:.0f}× ({LANG_META[l]['name']})" for w, l in pw)
                      + " as much as one unweighted word of general text"),
        n_fourth=sum(1 for r in fourth if r["lang"] != "-"), ur_level=lv["ur"], pa_level=lv["pa"], mr_level=lv["mr"],
        ur_words=manifest["pages"]["ur"]["words"], te_words=manifest["pages"]["te"]["words"],
        te_mult=final["weights"]["te.eval"] / w_en, n_runs=X["n_runs"], n_distinct=X["n_distinct_tokenizers"],
        n_recipe_runs=len(X["trace"]), final_max=F["max_fertility"])
    assert abs(trade["points"][0]["spread"] - F["spread"]) < 1e-12, "trade-off x1 point is not the released tokenizer"
    rec["eval_downloads"] = [{"lang": l, "href": f"data/eval/{l}.txt", "words": per[l]["words"],
                              "sha": per[l]["file_sha256"]} for l in langs]
    rec["check_snippet"] = (
        "# pip install tokenizers   (0.20 or newer)\n"
        "from tokenizers import Tokenizer\n"
        "tok = Tokenizer.from_file(\"tokenizer.json\")\n"
        "for lang in [\"en\", \"hi\", \"te\", \"ur\"]:          # the four .txt files from this page\n"
        "    text = open(f\"{lang}.txt\", encoding=\"utf-8\").read()\n"
        "    print(lang, len(tok.encode(text).ids), len(text.split()))")
    assert min(p["max"] for p in X["trace"]) >= F["max_fertility"] - 1e-12, "minimax claim would be false"
    E = {
        "total_runs": len(recs),
        "intro_html": T.INTRO.format(n=X["n_runs"], d=X["n_distinct_tokenizers"]),
        "baseline": {"label": "plain BPE on the four pages, equal weights", "id": base["id"],
                     "per_lang": {l: S.fert(base, l) for l in langs}, "spread": base["result"]["spread"],
                     "score": base["result"]["score"], "note_html": T.BASELINE_NOTE.format(score=base["result"]["score"]),
                     "ext_only": {"id": base_ext["id"], "per_lang": {l: S.fert(base_ext, l) for l in langs},
                                  "spread": base_ext["result"]["spread"], "score": base_ext["result"]["score"]}},
        "fourth": {"rows": fourth, "note_html": T.FOURTH_NOTE.format(minw=S.MIN_PAGE_WORDS)},
        "tradeoff": {"points": trade["points"], "en12": trade["en12"],
                     "note_html": T.TRADEOFF_NOTE.format(reach_score=reach["score"], final_score=F["score"])},
        "ablations": {"rows": abl, "note_html": T.ABLATION_NOTE},
        "trace": {"points": X["trace"], "note_html": T.TRACE_NOTE},
        "sensitivity": {
            "rows": [{**v, "primary": v["key"] == "pinned"} for v in sens["variants"]],
            "note_html": T.SENS_NOTE.format(alt_delta=alt_delta),
        },
    }
    rec["experiments"] = E
    rec["repro"] = {"blocks": [
        {"title": "Check the score with the files from this page",
         "html": ("<p>Download <code>tokenizer.json</code> and the four evaluation texts (top of the page) into one "
                  "folder and run the snippet shown there. The texts' SHA-256 values (as printed by "
                  "<code>sha256sum</code>) are in the table above and in <code>data/eval/manifest.json</code>.</p>")},
        {"title": "Check the score from the repository",
         "code": "python verify/verify_python.py tokenizer/tokenizer.json data/eval en,hi,te,ur"},
        {"title": "Full verification (Python + independent JavaScript + Unicode checks)",
         "code": "python verify/run_verification.py\npython -m pytest -q tests"},
        {"title": "Retrain the released tokenizer from scratch",
         "code": "python src/fetch_external.py en hi te ur     # extra Wikipedia text (HF wikimedia/wikipedia 20231101)\n"
                 "python src/make_final.py                    # trains with tokenizer/final_config.json and saves tokenizer.json"},
        {"title": "Software used",
         "html": (f"<p>Python {summ['versions']['python']} (Unicode {summ['versions']['python_unicode']}), Hugging Face "
                  f"tokenizers {summ['versions']['tokenizers']}, Node {summ['versions']['node']} (ICU Unicode "
                  f"{summ['versions']['node_icu_unicode']}). tokenizer.json SHA-256 <code>{sha}</code>. Extra training text: "
                  f"<code>wikimedia/wikipedia</code> 20231101 dump (dataset revision "
                  f"<code>b04c8d1ceb2f5cd4588862100d08de323dccfbaa</code>).</p>")},
    ]}
    (SITE / "data" / "site_data.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    bust_caches()
    print("site data written:", SITE / "data" / "site_data.json")


if __name__ == "__main__":
    main()
