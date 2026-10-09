"""Download additional (non-evaluation) Wikipedia text for tokenizer training.

Source: Hugging Face dataset `wikimedia/wikipedia`, config 20231101.<lang>, pinned to a
fixed dataset revision so the sample is reproducible. Articles are streamed in dataset
order and assigned to a split by a hash of the title:
    sha1(title) % 10 == 0  -> heldout (never used for training; used to measure generalization)
    otherwise              -> train pool
The India article of each language is skipped entirely (it is the evaluation page and
enters training, if at all, only through the pinned evaluation snapshot).
"""
import gzip
import hashlib
import json
import pathlib
import sys
import time

from datasets import load_dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "external"
DATASET = "wikimedia/wikipedia"
DUMP = "20231101"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
TRAIN_WORDS = 600_000
HELDOUT_WORDS = 60_000

sys.path.insert(0, str(ROOT / "src"))
from fetch_india import TITLES  # noqa: E402


def split_of(title):
    return "heldout" if int(hashlib.sha1(title.encode("utf-8")).hexdigest(), 16) % 10 == 0 else "train"


def fetch(lang):
    OUT.mkdir(parents=True, exist_ok=True)
    india = {TITLES.get(lang), "India", "भारत", "Bharat"}
    ds = load_dataset(DATASET, f"{DUMP}.{lang}", split="train", streaming=True, revision=REVISION)
    words = {"train": 0, "heldout": 0}
    files = {s: gzip.open(OUT / f"{lang}.{s}.jsonl.gz", "wt", encoding="utf-8") for s in words}
    caps = {"train": TRAIN_WORDS, "heldout": HELDOUT_WORDS}
    n = 0
    t0 = time.time()
    for ex in ds:
        if ex["title"] in india:
            continue
        s = split_of(ex["title"])
        if words[s] >= caps[s]:
            if all(words[k] >= caps[k] for k in caps):
                break
            continue
        w = len(ex["text"].split())
        if w < 20:
            continue
        files[s].write(json.dumps({"id": ex["id"], "title": ex["title"], "text": ex["text"]}, ensure_ascii=False) + "\n")
        words[s] += w
        n += 1
    for f in files.values():
        f.close()
    print(f"{lang}: docs={n} train_words={words['train']} heldout_words={words['heldout']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for lang in sys.argv[1:]:
        fetch(lang)
