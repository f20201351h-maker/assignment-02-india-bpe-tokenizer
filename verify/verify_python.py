"""Independent re-computation of the assignment numbers from the exported artifact only.

Imports nothing from src/. Inputs: tokenizer.json + the pinned evaluation snapshots.
    python verify/verify_python.py tokenizer/tokenizer.json data/eval en,hi,te,mr [out.json]

Fertility = tokens / words for each page; words = whitespace-separated chunks.
Spread and score are computed with exact fractions, then printed at full precision.
"""
import hashlib
import json
import platform
import re
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer


def words_split(text):
    return len(text.split())


def words_regex(text):  # second, independent way of counting the same thing
    return len(re.findall(r"\S+", text))


def exact_decimal(x: Fraction, digits: int) -> str:
    """Decimal expansion of a fraction, truncated (not rounded) to `digits` places, no float involved."""
    q = x.numerator * 10**digits // x.denominator
    return f"{q // 10**digits}.{q % 10**digits:0{digits}d}"


def main(tok_path, eval_dir, langs, out_path=None):
    eval_dir = Path(eval_dir)
    raw = Path(tok_path).read_bytes()
    tj = json.loads(raw)
    tok = Tokenizer.from_file(str(tok_path))
    manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in tj["model"]["merges"]]
    rep = {
        "implementation": "verify/verify_python.py (Hugging Face tokenizers, loaded from tokenizer.json only)",
        "python": platform.python_version(), "tokenizers": tokenizers.__version__,
        "unicode_version": unicodedata.unidata_version,
        "tokenizer_sha256": hashlib.sha256(raw).hexdigest(),
        "model_type": tj["model"]["type"],
        "vocab_size_hf": tok.get_vocab_size(with_added_tokens=True),
        "vocab_entries_in_json": len(vocab),
        "added_tokens": [a["content"] for a in tj.get("added_tokens", [])],
        "vocab_ids_contiguous": sorted(vocab.values()) == list(range(len(vocab))),
        "n_merges": len(merges),
        "merges_valid": all(a in vocab and b in vocab and a + b in vocab for a, b in merges),
        "post_processor": tj.get("post_processor"),
        "languages": {},
    }
    fert = {}
    for lang in langs:
        raw_bytes = (eval_dir / f"{lang}.txt").read_bytes()
        text = raw_bytes.decode("utf-8").rstrip("\n")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        meta = manifest["pages"][lang]
        enc = tok.encode(text)
        w1, w2 = words_split(text), words_regex(text)
        n_tok = len(enc.ids)
        fert[lang] = Fraction(n_tok, w1)
        norm = tok.normalizer.normalize_str(text)
        rep["languages"][lang] = {
            "language": meta["language"], "title": meta["title"], "revid": meta["revid"],
            "clean_text_sha256": sha, "sha_matches_manifest": sha == meta["clean_text_sha256"],
            "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "file_sha_matches_manifest": hashlib.sha256(raw_bytes).hexdigest() == meta.get("file_sha256"),
            "tokens_untrimmed_file": len(tok.encode(raw_bytes.decode("utf-8")).ids),
            "tokens_line_by_line": sum(len(tok.encode(line).ids) for line in text.split("\n")),
            "words": w1, "words_regex_check": w2, "tokens": n_tok,
            "tokens_with_special": len(tok.encode(text, add_special_tokens=True).ids),
            "fertility": float(fert[lang]), "fertility_exact": f"{n_tok}/{w1}",
            "unk_tokens": enc.ids.count(tok.token_to_id("<unk>")) if tok.token_to_id("<unk>") is not None else 0,
            "roundtrip_exact": tok.decode(enc.ids) == norm,
            "nfd_input_same_tokens": tok.encode(unicodedata.normalize("NFD", text)).ids == enc.ids,
            "deterministic": tok.encode(text).ids == enc.ids,
        }
    lo = min(fert, key=fert.get)
    hi = max(fert, key=fert.get)
    spread = fert[hi] - fert[lo]
    rep.update({
        "sorted": sorted(((l, float(f)) for l, f in fert.items()), key=lambda x: x[1]),
        "min_lang": lo, "max_lang": hi,
        "min_fertility": float(fert[lo]), "max_fertility": float(fert[hi]),
        "spread": float(spread), "spread_exact": f"{spread.numerator}/{spread.denominator}",
        "score": float(Fraction(1000) / spread) if spread else None,
        "score_exact": (lambda s: f"{s.numerator}/{s.denominator}")(Fraction(1000) / spread) if spread else None,
        "score_decimal_30": exact_decimal(Fraction(1000) / spread, 12) if spread else None,
    })
    js = json.dumps(rep, ensure_ascii=False, indent=1)
    if out_path:
        Path(out_path).write_text(js, encoding="utf-8", newline="\n")
    print(js)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4] if len(sys.argv) > 4 else None)
