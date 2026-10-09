"""Alternative extractions of the SAME revisions, used only for a sensitivity check.

  html_prose : paragraphs, headings and list items from the rendered HTML, with tables,
               infobox, references, figures and navigation boxes removed (close to TextExtracts).
  html_all   : every piece of visible text inside the article body, including the infobox,
               tables, image captions, reference list and navigation boxes (a "naive" scrape).
Both are cleaned with the same textnorm.clean_text as the evaluation snapshots.
"""
import gzip
import json
import pathlib
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from textnorm import clean_text, count_words  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "india"
OUT = ROOT / "data" / "alt"

DROP_PROSE = ["style", "script", "sup.reference", ".mw-editsection", "table", "figure", ".thumb",
              ".navbox", ".reflist", "ol.references", ".infobox", ".metadata", ".mw-empty-elt",
              ".noprint", ".gallery", "math", ".mwe-math-element"]


def _root(html):
    s = BeautifulSoup(html, "html.parser")
    return s.find(class_="mw-parser-output") or s


def html_prose(html):
    root = _root(html)
    for sel in DROP_PROSE:
        for x in root.select(sel):
            x.decompose()
    parts = [el.get_text("") for el in root.find_all(["p", "li", "h2", "h3", "h4", "h5"])
             if not el.find_parent(["li"]) or el.name == "li"]
    return clean_text("\n".join(parts))


def html_all(html):
    root = _root(html)
    for x in root.select("style, script"):
        x.decompose()
    return clean_text(root.get_text("\n"))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(RAW.glob("*.html.gz")):
        lang = f.name.split(".")[0]
        html = gzip.open(f, "rt", encoding="utf-8").read()
        for name, fn in [("html_prose", html_prose), ("html_all", html_all)]:
            t = fn(html)
            (OUT / f"{lang}.{name}.txt").write_text(t + "\n", encoding="utf-8", newline="\n")
        print(lang, "prose", count_words((OUT / f"{lang}.html_prose.txt").read_text(encoding="utf-8")),
              "all", count_words((OUT / f"{lang}.html_all.txt").read_text(encoding="utf-8")))
