"""Fetch the 'India' Wikipedia article in each candidate language (raw API output).

Writes data/raw/india/<lang>.json (TextExtracts text + revision metadata) and
data/raw/india/<lang>.html.gz (rendered HTML of the same revision, used only for
a sensitivity check with a different extraction method).
"""
import datetime as dt
import gzip
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wiki import fetch_extract, fetch_html  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "india"

# Titles of the India article, taken from en:India's interlanguage links.
TITLES = {
    "en": "India", "hi": "भारत", "te": "భారతదేశం",
    # fourth-language candidates
    "mr": "भारत", "ne": "भारत", "sa": "भारतम्", "gom": "भारत", "mai": "भारत",
    "bn": "ভারত", "as": "ভাৰত", "kn": "ಭಾರತ", "ta": "இந்தியா", "ml": "ഇന്ത്യ",
    "gu": "ભારત", "pa": "ਭਾਰਤ", "or": "ଭାରତ", "ur": "بھارت",
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    langs = sys.argv[1:] or list(TITLES)
    for lang in langs:
        rec = fetch_extract(lang, TITLES[lang])
        rec["retrieved_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["source"] = "MediaWiki Action API: action=query&prop=extracts&explaintext=1&exsectionformat=plain"
        html = fetch_html(lang, rec["revid"])
        (OUT / f"{lang}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        with gzip.open(OUT / f"{lang}.html.gz", "wt", encoding="utf-8") as f:
            f.write(html)
        print(f"{lang:4s} rev={rec['revid']} {rec['title']} chars={len(rec['text'])} words={len(rec['text'].split())}")
