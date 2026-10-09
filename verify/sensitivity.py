"""How much do the fertilities/score move if the page text is obtained differently?

Uses only tokenizer.json. Variants:
  pinned      : the frozen evaluation snapshots (the reported result)
  html_prose  : same revisions, paragraphs/headings/lists taken from the rendered HTML
  html_all    : same revisions, every visible text node of the article body (infobox, tables,
                captions, references, navboxes included) - a careless scrape
  current     : the live article fetched now through the same TextExtracts API (optional, --live)
usage: python verify/sensitivity.py tokenizer.json en,hi,te,ur out.json [--live]
"""
import datetime as dt
import json
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def clean(raw):  # same rule as src/textnorm.clean_text, re-typed here on purpose
    t = unicodedata.normalize("NFC", raw)
    brk = "\n\r  \x0b\x0c\x1c\x1d\x1e\x85"
    t = "".join("\n" if c in brk else (" " if c.isspace() else c) for c in t)
    lines = [" ".join(l.split()) for l in t.split("\n")]
    return "\n".join(l for l in lines if l)


def measure(tok, texts):
    f = {l: Fraction(len(tok.encode(t).ids), len(t.split())) for l, t in texts.items()}
    s = max(f.values()) - min(f.values())
    return {"per_lang": {l: float(v) for l, v in f.items()},
            "words": {l: len(t.split()) for l, t in texts.items()},
            "spread": float(s), "score": float(1000 / s) if s else None}


def fetch_current(langs, sectionformat="plain"):
    """sectionformat="wiki" is the TextExtracts default ("== Heading ==" lines), which is what the
    popular `wikipedia` Python package returns as page.content."""
    import requests
    manifest = json.loads((ROOT / "data" / "eval" / "manifest.json").read_text(encoding="utf-8"))
    out, meta = {}, {}
    s = requests.Session()
    s.headers["User-Agent"] = "india-bpe-tokenizer/1.0 (educational tokenizer project; python-requests)"
    for l in langs:
        title = manifest["pages"][l]["title"]
        r = s.get(f"https://{l}.wikipedia.org/w/api.php", params={
            "action": "query", "prop": "extracts|revisions", "titles": title, "explaintext": 1,
            "exsectionformat": sectionformat, "rvprop": "ids|timestamp", "format": "json", "formatversion": 2},
            timeout=60)
        p = r.json()["query"]["pages"][0]
        out[l] = clean(p["extract"])
        meta[l] = {"revid": p["revisions"][0]["revid"], "timestamp": p["revisions"][0]["timestamp"],
                   "same_as_pinned": p["revisions"][0]["revid"] == manifest["pages"][l]["revid"]}
        import time
        time.sleep(1.0)
    return out, meta


if __name__ == "__main__":
    tok_path, langs, out_path = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    tok = Tokenizer.from_file(tok_path)
    res = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "variants": []}
    pinned = {l: (ROOT / "data" / "eval" / f"{l}.txt").read_text(encoding="utf-8").rstrip("\n") for l in langs}
    res["variants"].append({"variant": "Pinned snapshot (reported)", "key": "pinned",
                            "desc": "TextExtracts plain text of the pinned revisions", **measure(tok, pinned)})
    for key, label, desc in [
        ("html_prose", "HTML prose, same revisions", "paragraphs, headings and list items from the rendered HTML"),
        ("html_all", "Whole page scrape, same revisions", "all visible text incl. infobox, tables, captions, references"),
    ]:
        texts = {l: (ROOT / "data" / "alt" / f"{l}.{key}.txt").read_text(encoding="utf-8").rstrip("\n") for l in langs}
        res["variants"].append({"variant": label, "key": key, "desc": desc, **measure(tok, texts)})
    if "--live" in sys.argv:
        cur, meta = fetch_current(langs)
        changed = [l for l in langs if not meta[l]["same_as_pinned"]]
        res["variants"].append({"variant": "Live article today", "key": "current",
                                "desc": ("current revisions fetched " + res["generated_utc"][:10] + "; edited since pinning: "
                                         + (", ".join(changed) if changed else "none")),
                                "revisions": meta, **measure(tok, cur)})
        cur_w, meta_w = fetch_current(langs, sectionformat="wiki")
        res["variants"].append({"variant": "Live article, '== Heading ==' format", "key": "current_wiki",
                                "desc": ("TextExtracts default section format, as returned by the `wikipedia` Python "
                                         "package; headings become '== Title ==' so each '==' counts as a word"),
                                "revisions": meta_w, **measure(tok, cur_w)})
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(res, ensure_ascii=False, indent=1))
