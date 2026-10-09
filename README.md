# India BPE tokenizer

Built as ERA V5 Session 2 assignment.

A single **Byte-Pair-Encoding tokenizer with exactly 10,000 tokens** for the Wikipedia article
on India in **English, Hindi, Telugu and Urdu** (Urdu = my fourth language, chosen by experiment).

| Language | Page (pinned revision) | Words | Tokens | Fertility |
|---|---|---:|---:|---:|
| English | [India](https://en.wikipedia.org/w/index.php?oldid=1377830907) (1377830907) | 11,672 | 17,292 | 1.481494 |
| Hindi | [भारत](https://hi.wikipedia.org/w/index.php?oldid=6622959) (6622959) | 8,187 | 12,131 | 1.481739 |
| Telugu | [భారతదేశం](https://te.wikipedia.org/w/index.php?oldid=4848340) (4848340) | 2,453 | 3,634 | 1.481451 |
| Urdu | [بھارت](https://ur.wikipedia.org/w/index.php?oldid=11983228) (11983228) | 3,486 | 5,166 | 1.481928 |

* Best (X_min): **Telugu 1.481451** · worst (X_max): **Urdu 1.481928**
* Spread X_max − X_min = **97/203599 = 0.000476426701506**
* **Score = 1000 / spread = 2098958.762886597938**
* Vocabulary: **10,000** = 1 special (`<unk>`) + 365 single characters
  + 1406 akshara-building merges + 8,228 learned merges. Model type BPE,
  no added tokens, no post-processor.
* Tokenizer: [`tokenizer/tokenizer.json`](tokenizer/tokenizer.json)
  (SHA-256 `45aa8369f867472f8686b48025354fbfc0d9b53d6f7573d021aaaac815179dc4`); token list: [`site/tokens.tsv`](site/tokens.tsv).
* All numbers above were re-computed from the exported `tokenizer.json` by two independent
  implementations (Hugging Face `tokenizers` in Python and a separate JavaScript BPE):
  **56/56 verification checks pass** (`verify/out/verification_summary.json`).

Fertility = tokens on the page ÷ words on the page. Words = whitespace-separated chunks of the
cleaned page text (`len(text.split())`), the rule from the course's worked example ("Artificial
intelligence is changing the world" = 6 words).

**Honest caveats**

1. English fertility (1.481) is above the 1.2 target set for the task (it is within
   the looser 1.6 bound). English ≤ 1.2 is reachable: with 6× the English weight it is
   1.189, but Telugu then needs 2.348 tokens per word and the score falls to
   863. With tokens that never cross a space I could not get all four to 1.2
   together: English, Hindi and Telugu alone already meet at 1.426, and a balancing run with
   English pinned near 1.2 (best attempt 1.208) left the other three at about
   2.08 (score 1,141). See `experiments/RESEARCH_LOG.md`.
2. The four evaluation pages are part of the tokenizer's training text (the task allows any
   data for building the tokenizer). Fertility on unseen Wikipedia text is higher
   (held-out: English 2.08, Hindi 1.98, Telugu 2.74, Urdu 1.93).
3. The score is for the pinned page texts in `data/eval/`. Re-measured on the live pages
   (2026-10-01) in the same format it is unchanged. A different extraction of the same pages gives
   other fertilities (up to 0.59 different) and a different score: about 93,821 with
   the "== Heading ==" format returned by the `wikipedia` Python package, a few thousand when the
   whole HTML page is scraped (`verify/out/sensitivity.json`). English is only 0.5 token
   above the best language, so even one token more or less on a page changes the score a lot.
4. Characters that never occurred in training (e.g. most Telugu and Urdu digits, emoji, other scripts)
   become `<unk>` (one token per character). None occur on the four pages; on ~244K words of
   held-out text there were 357 such tokens. Decoding returns the text with all whitespace
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
    text = open(f"data/eval/{lang}.txt", encoding="utf-8").read()
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
python src/assemble_experiments.py C-akfew && python src/build_site.py && python src/write_readme.py
```

Note: `src/fetch_india.py` fetches the *current* revisions. The pinned texts used for the score
are already in `data/eval/` (with revision ids and SHA-256 in `data/eval/manifest.json`), so
re-running it is only needed to re-measure on today's pages.

## How it was built (short)

* **Pages**: MediaWiki Action API, TextExtracts plain text (Wikimedia strips tables, infoboxes,
  references and navigation the same way for every language). Cleaning: NFC + whitespace only.
* **Pre-tokenization**: SentencePiece-style — a space becomes "▁" on the following word, so decoding
  is exact; tokens never cross a space. Splitting punctuation off was tried and rejected: it makes
  1.227 the lowest possible English fertility on this page.
* **Akshara-first BPE**: the first merges glue each akshara (consonant + vowel signs, virama
  conjuncts such as क्ष / క్ష, ZWJ kept inside, ZWNJ respected as a break); learned merges only join
  whole aksharas. Zero token boundaries inside an akshara on all four pages (checked with my own
  rule and independently with ICU grapheme segmentation). Cost ≈ 0.026 fertility vs plain BPE.
* **Training text**: the four pages ×300 + 600,000 words of other Wikipedia articles per
  language; per-language sampling weights tuned so that the four fertilities meet.
* **Fourth language**: 10 candidates trained with the same recipe; Urdu gave the lowest
  balanced fertility. See `experiments/SUMMARY.md`.
* **Search**: 519 training runs (`experiments/results.jsonl`,
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
