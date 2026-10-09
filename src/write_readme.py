"""Render README.md from the verification outputs (so the numbers in it cannot drift)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "ur": "Urdu"}


def j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


TEMPLATE = """# India BPE tokenizer

Built as ERA V5 Session 2 assignment.

A single **Byte-Pair-Encoding tokenizer with exactly 10,000 tokens** for the Wikipedia article
on India in **English, Hindi, Telugu and Urdu** (Urdu = my fourth language, chosen by experiment).

| Language | Page (pinned revision) | Words | Tokens | Fertility |
|---|---|---:|---:|---:|
{rows}

* Best (X_min): **{min_name} {min_f:.6f}** · worst (X_max): **{max_name} {max_f:.6f}**
* Spread X_max − X_min = **{spread_exact} = {spread:.12g}**
* **Score = 1000 / spread = {score_decimal}**
* Vocabulary: **{vocab:,}** = {comp_special} special (`<unk>`) + {comp_chars} single characters
  + {comp_ak} akshara-building merges + {comp_learned:,} learned merges. Model type BPE,
  no added tokens, no post-processor.
* Tokenizer: [`tokenizer/tokenizer.json`](tokenizer/tokenizer.json)
  (SHA-256 `{sha}`); token list: [`site/tokens.tsv`](site/tokens.tsv).
* All numbers above were re-computed from the exported `tokenizer.json` by two independent
  implementations (Hugging Face `tokenizers` in Python and a separate JavaScript BPE):
  **{n_pass}/{n_checks} verification checks pass** (`verify/out/verification_summary.json`).

Fertility = tokens on the page ÷ words on the page. Words = whitespace-separated chunks of the
cleaned page text (`len(text.split())`), the rule from the course's worked example ("Artificial
intelligence is changing the world" = 6 words).

**Honest caveats**

1. English fertility ({en_f:.3f}) is above the 1.2 target set for the task (it is within
   the looser 1.6 bound). English ≤ 1.2 is reachable: with 6× the English weight it is
   {reach_en:.3f}, but Telugu then needs {reach_worst:.3f} tokens per word and the score falls to
   {reach_score:,.0f}. With tokens that never cross a space I could not get all four to 1.2
   together: English, Hindi and Telugu alone already meet at {lvl3:.3f}, and a balancing run with
   English pinned near 1.2 (best attempt {en12_en:.3f}) left the other three at about
   {en12_worst:.2f} (score {en12_score:,.0f}). See `experiments/RESEARCH_LOG.md`.
2. The four evaluation pages are part of the tokenizer's training text (the task allows any
   data for building the tokenizer). Fertility on unseen Wikipedia text is higher
   (held-out: {heldout}).
3. The score is for the pinned page texts in `data/eval/`. Re-measured on the live pages
   ({live_date}) in the same format it is {live_note}. A different extraction of the same pages gives
   other fertilities (up to {alt_delta:.2f} different) and a different score: about {wiki_score:,.0f} with
   the "== Heading ==" format returned by the `wikipedia` Python package, a few thousand when the
   whole HTML page is scraped (`verify/out/sensitivity.json`). English is only {en_margin:.1f} token
   above the best language, so even one token more or less on a page changes the score a lot.
4. Characters that never occurred in training (e.g. most Telugu and Urdu digits, emoji, other scripts)
   become `<unk>` (one token per character). None occur on the four pages; on ~244K words of
   held-out text there were {heldout_unk} such tokens. Decoding returns the text with all whitespace
   collapsed to single spaces.

## Check the score yourself

```bash
pip install "tokenizers>=0.20"     # tested with 0.20.3, 0.21.0 and 0.23.1 (identical counts)
python verify/verify_python.py tokenizer/tokenizer.json data/eval en,hi,te,ur
```

or, with nothing but `tokenizer.json` and the four text files in `data/eval/`:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")
for lang in ["en", "hi", "te", "ur"]:
    text = open(f"data/eval/{{lang}}.txt", encoding="utf-8").read()
    print(lang, len(tok.encode(text).ids) / len(text.split()))
```

Full verification (Python + independent JavaScript + Unicode checks) and tests:

```bash
python verify/run_verification.py
python -m pytest -q tests
```

On a Windows console, set `PYTHONIOENCODING=utf-8` first; the verification report prints Indic and Urdu text.

## Reproduce everything from scratch

```bash
pip install -r requirements.txt
python src/fetch_india.py                 # India article in 17 languages via the MediaWiki API (raw JSON + HTML)
python src/make_snapshots.py              # cleaned evaluation texts + data/eval/manifest.json
python src/fetch_external.py en hi te ur  # extra training text (HF wikimedia/wikipedia 20231101, pinned revision)
python src/make_final.py                  # trains the released tokenizer from tokenizer/final_config.json
python src/make_final.py --check          # retrains and confirms the result is byte-identical
python verify/run_verification.py
python verify/sensitivity.py tokenizer/tokenizer.json en,hi,te,ur verify/out/sensitivity.json --live
python src/assemble_experiments.py {final_stage} && python src/build_site.py && python src/write_readme.py
```

Note: `src/fetch_india.py` fetches the *current* revisions. The pinned texts used for the score
are already in `data/eval/` (with revision ids and SHA-256 in `data/eval/manifest.json`), so
re-running it is only needed to re-measure on today's pages.

## How it was built (short)

* **Pages**: MediaWiki Action API, TextExtracts plain text (Wikimedia strips tables, infoboxes,
  references and navigation the same way for every language). Cleaning: NFC + whitespace only.
* **Pre-tokenization**: SentencePiece-style — a space becomes "▁" on the following word, so decoding
  is exact; tokens never cross a space. Splitting punctuation off was tried and rejected: it makes
  {punct:.3f} the lowest possible English fertility on this page.
* **Akshara-first BPE**: the first merges glue each akshara (consonant + vowel signs, virama
  conjuncts such as क्ष / క్ష, ZWJ kept inside, ZWNJ respected as a break); learned merges only join
  whole aksharas. Zero token boundaries inside an akshara on all four pages (checked with my own
  rule and independently with ICU grapheme segmentation). Cost ≈ {ak_cost:.3f} fertility vs plain BPE.
* **Training text**: the four pages ×{eval_w:g} + {ext_words:,} words of other Wikipedia articles per
  language; per-language sampling weights tuned so that the four fertilities meet.
* **Fourth language**: {n_fourth} candidates trained with the same recipe; Urdu gave the lowest
  balanced fertility. See `experiments/SUMMARY.md`.
* **Search**: {n_runs} training runs (`experiments/results.jsonl`,
  `experiments/ledger/*.jsonl`), decisions in `experiments/RESEARCH_LOG.md`.

## Repository layout

```
data/eval/            pinned evaluation texts + manifest (revision ids, hashes, word counts)
data/raw/india/       raw API responses (TextExtracts JSON, rendered HTML) for 17 languages
data/alt/             alternative extractions used only for the sensitivity check
src/                  fetching, cleaning, akshara rule, training pipeline, search, site builder
tokenizer/            tokenizer.json (the released tokenizer) + final_config.json (exact recipe/weights)
verify/               independent verification (Python, JavaScript), sensitivity, outputs
experiments/          ledgers of every run, research log, generated summary
site/                 the website (static; Netlify publish directory)
tests/                pytest suite
```

## Licence / attribution

Evaluation and training text comes from Wikipedia (CC BY-SA 4.0); page authors are listed in each
page's history (links in `data/eval/manifest.json`). Extra training text: the `wikimedia/wikipedia`
dataset on Hugging Face (same licence).
"""


def main():
    s = j("verify/out/verification_summary.json")
    final = j("tokenizer/final_config.json")
    X = j("experiments/site_experiments.json")
    sens = j("verify/out/sensitivity.json")
    site = j("site/data/site_data.json")
    man = j("data/eval/manifest.json")["pages"]
    rows = "\n".join(
        f"| {NAMES[l]} | [{man[l]['title']}]({man[l]['permanent_url']}) ({man[l]['revid']}) | {s['words'][l]:,} | "
        f"{s['tokens'][l]:,} | {s['fertility'][l]:.6f} |" for l in s["languages"])
    fourth = [r for r in site["experiments"]["fourth"]["rows"]]
    lvl3 = next(r["level"] for r in fourth if r["lang"] == "-")
    plain = next(a for a in X["ablations"] if a["stage"] == "C-plain2")
    chosen = next(a for a in X["ablations"] if a["chosen"])
    c = final["composition"]
    text = TEMPLATE.format(
        rows=rows, min_name=NAMES[s["min_lang"]], min_f=s["fertility"][s["min_lang"]],
        max_name=NAMES[s["max_lang"]], max_f=s["fertility"][s["max_lang"]],
        spread_exact=s["spread_exact"], spread=s["spread"], score_decimal=s["score_decimal"],
        vocab=10000, comp_special=c["special"], comp_chars=c["characters"], comp_ak=c["akshara"],
        comp_learned=c["learned"], sha=s["tokenizer_sha256"],
        n_pass=sum(x["pass"] for x in s["checks"]), n_checks=len(s["checks"]),
        en_f=s["fertility"]["en"], lvl3=lvl3, en12_worst=X["tradeoff"]["en12"]["worst"], en12_en=X["tradeoff"]["en12"]["en"],
        en12_score=X["tradeoff"]["en12"]["score"],
        heldout=", ".join(f"{NAMES[l]} {final['heldout'][l]['fertility']:.2f}" for l in s["languages"]),
        sens="; ".join(f"{v['variant']}: spread {v['spread']:.3g}, score {v['score']:,.0f}" for v in sens["variants"]),
        live_date=sens["generated_utc"][:10],
        live_note=next(("unchanged" if abs(v["spread"] - sens["variants"][0]["spread"]) < 1e-12 else f"{v['score']:,.0f}")
                       for v in sens["variants"] if v["key"] == "current"),
        alt_delta=max(abs(v["per_lang"][l] - sens["variants"][0]["per_lang"][l]) for v in sens["variants"]
                      if v["key"] in ("html_prose", "html_all") for l in s["languages"]),
        punct=X["punct_floor_en"], ak_cost=chosen["max"] - plain["max"], eval_w=final["eval_w"],
        ext_words=final["config"]["ext_words"], n_fourth=sum(1 for r in fourth if r["lang"] != "-"),
        n_runs=site["experiments"]["total_runs"], final_stage=final["source_stage"],
        reach_en=X["tradeoff"]["first_en_le_12"]["en"], reach_worst=X["tradeoff"]["first_en_le_12"]["worst_other"],
        reach_score=X["tradeoff"]["first_en_le_12"]["score"],
        wiki_score=next(v["score"] for v in sens["variants"] if v["key"] == "current_wiki"),
        en_margin=(s["fertility"]["en"] - s["fertility"][s["min_lang"]]) * s["words"]["en"],
        heldout_unk=sum(final["heldout"][l]["unk"] for l in s["languages"]))
    (ROOT / "README.md").write_text(text, encoding="utf-8", newline="\n")
    print("README.md written")


if __name__ == "__main__":
    main()
