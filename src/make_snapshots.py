"""Freeze the evaluation pages: cleaned text + manifest (revision ids, hashes, counts)."""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from textnorm import clean_text, count_words  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "india"
OUT = ROOT / "data" / "eval"

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi", "bn": "Bengali",
    "ta": "Tamil", "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "ur": "Urdu",
    "kn": "Kannada", "gu": "Gujarati", "sa": "Sanskrit", "ne": "Nepali", "as": "Assamese",
    "gom": "Konkani", "mai": "Maithili",
}


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "extraction_policy": (
            "MediaWiki Action API TextExtracts plain text (action=query&prop=extracts&explaintext=1"
            "&exsectionformat=plain) of the current revision at retrieval time; Wikimedia strips tables, "
            "infoboxes, figures, reference lists and navigation server-side, identically for every language. "
            "Then src/textnorm.clean_text: NFC, all whitespace -> single spaces, line breaks kept between "
            "paragraphs, empty lines dropped. Headings and list items are kept as lines."
        ),
        "word_count_rule": "len(text.split()) on the cleaned text (whitespace-separated chunks)",
        "license": "Text from Wikipedia, CC BY-SA 4.0; see each page URL for authors/history.",
        "pages": {},
    }
    for f in sorted(RAW.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        lang = rec["lang"]
        text = clean_text(rec["text"])
        (OUT / f"{lang}.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
        manifest["pages"][lang] = {
            "language": LANG_NAMES.get(lang, lang), "project": f"{lang}.wikipedia.org",
            "title": rec["title"], "pageid": rec["pageid"], "revid": rec["revid"],
            "revision_timestamp": rec["rev_timestamp"], "retrieved_utc": rec["retrieved_utc"],
            "url": rec["url"],
            "permanent_url": f"https://{lang}.wikipedia.org/w/index.php?oldid={rec['revid']}",
            "raw_extract_sha256": sha256(rec["text"]),
            "clean_text_sha256": sha256(text),            # hash of the text (file content without the final newline)
            "file_sha256": sha256(text + "\n"),           # hash of the file bytes, as shown by `sha256sum`
            "chars": len(text), "words": count_words(text), "lines": text.count("\n") + 1,
        }
        print(lang, manifest["pages"][lang]["words"], manifest["pages"][lang]["clean_text_sha256"][:12])
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
