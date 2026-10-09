"""Tests for the released tokenizer, the evaluation data and the website data.

Run:  python -m pytest -q tests
(The retraining test takes about a minute; skip it with  -m "not slow".)
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path

import pytest
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from akshara import aksharas, bad_splits  # noqa: E402
from textnorm import clean_text, count_words  # noqa: E402

TOK_PATH = ROOT / "tokenizer" / "tokenizer.json"
FINAL = json.loads((ROOT / "tokenizer" / "final_config.json").read_text(encoding="utf-8"))
LANGS = FINAL["langs"]
MANIFEST = json.loads((ROOT / "data" / "eval" / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_file(str(TOK_PATH))


@pytest.fixture(scope="module")
def tj():
    return json.loads(TOK_PATH.read_text(encoding="utf-8"))


def page(lang):
    return (ROOT / "data" / "eval" / f"{lang}.txt").read_text(encoding="utf-8").rstrip("\n")


# ------------------------------------------------------------------ artifact
def test_vocab_size_is_exactly_10000(tok, tj):
    assert tok.get_vocab_size(with_added_tokens=True) == 10_000
    assert len(tj["model"]["vocab"]) == 10_000
    assert sorted(tj["model"]["vocab"].values()) == list(range(10_000))


def test_is_bpe_with_valid_merges(tj):
    assert tj["model"]["type"] == "BPE"
    v = tj["model"]["vocab"]
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in tj["model"]["merges"]]
    assert all(a in v and b in v and a + b in v for a, b in merges)
    # every non-base token is produced by exactly one merge
    produced = [a + b for a, b in merges]
    assert len(produced) == len(set(produced))
    singles = [t for t in v if len(t) == 1 or t == "<unk>"]
    assert len(singles) + len(merges) == 10_000
    assert tj.get("post_processor") is None
    assert not tj["model"].get("ignore_merges")


def test_reload_is_deterministic(tok):
    t2 = Tokenizer.from_file(str(TOK_PATH))
    for l in LANGS:
        assert tok.encode(page(l)).ids == t2.encode(page(l)).ids


# ------------------------------------------------------------------ evaluation data
@pytest.mark.parametrize("lang", LANGS)
def test_snapshot_hash_and_cleaning(lang):
    text = page(lang)
    meta = MANIFEST["pages"][lang]
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == meta["clean_text_sha256"]
    assert clean_text(text) == text                      # snapshot is already in clean form
    assert unicodedata.is_normalized("NFC", text)
    raw = json.loads((ROOT / "data" / "raw" / "india" / f"{lang}.json").read_text(encoding="utf-8"))
    assert raw["revid"] == meta["revid"]
    assert clean_text(raw["text"]) == text               # snapshot regenerates from the raw API output


@pytest.mark.parametrize("lang", LANGS)
def test_word_count_rule_consistent(lang):
    text = page(lang)
    n = count_words(text)
    assert n == len(text.split()) == len(re.findall(r"\S+", text)) == MANIFEST["pages"][lang]["words"]


def test_word_count_examples():
    assert count_words("Artificial intelligence is changing the world") == 6   # lecture example
    assert count_words("भारत दक्षिण एशिया में स्थित है।") == 6
    assert count_words("భారతదేశం దక్షిణ ఆసియాలో ఉంది.") == 4
    assert count_words("بھارت جنوبی ایشیا میں واقع ایک ملک ہے۔") == 8
    assert count_words("a b c  d\n\ne") == 5         # NBSP / thin space / newlines separate words
    assert count_words("जवाहर‌लाल") == 1                  # ZWNJ is not a word separator


# ------------------------------------------------------------------ fertility / score
def test_fertility_and_score_recomputed(tok):
    f = {l: Fraction(len(tok.encode(page(l)).ids), count_words(page(l))) for l in LANGS}
    spread = max(f.values()) - min(f.values())
    summ = json.loads((ROOT / "verify" / "out" / "verification_summary.json").read_text(encoding="utf-8"))
    assert summ["all_pass"]
    for l in LANGS:
        assert f"{f[l].numerator * (summ['words'][l] // f[l].denominator)}/{summ['words'][l]}" == summ["fertility_exact"][l]
    assert f"{spread.numerator}/{spread.denominator}" == summ["spread_exact"]
    assert float(1000 / spread) == pytest.approx(summ["score"], rel=1e-12)
    assert summ["min_lang"] == min(f, key=f.get) and summ["max_lang"] == max(f, key=f.get)


def test_no_unknown_tokens_and_lossless(tok):
    for l in LANGS:
        text = page(l)
        enc = tok.encode(text)
        assert "<unk>" not in enc.tokens
        assert tok.decode(enc.ids) == tok.normalizer.normalize_str(text)


# ------------------------------------------------------------------ Indic / Unicode behaviour
def test_akshara_segmentation_examples():
    assert aksharas("शक्ति") == ["श", "क्ति"]
    assert aksharas("स्वतंत्रता") == ["स्व", "तं", "त्र", "ता"]
    assert aksharas("క్ష") == ["క్ష"]
    assert aksharas("స్వాతంత్ర్యం") == ["స్వా", "తం", "త్ర్యం"]
    assert aksharas("क्‍ष") == ["क्‍ष"]           # ZWJ keeps the conjunct together
    assert aksharas("क्‌ष") == ["क्‌", "ष"]       # ZWNJ asks for separate consonants
    assert aksharas("இந்தியா") == ["இ", "ந்", "தி", "யா"]    # Tamil pulli does not conjoin


@pytest.mark.parametrize("lang", LANGS)
def test_no_token_boundary_inside_an_akshara(tok, lang):
    enc = tok.encode(page(lang))
    groups = {}
    for t, w in zip(enc.tokens, enc.word_ids):
        groups.setdefault(w, []).append(t)
    assert sum(bad_splits("".join(p), p) for p in groups.values()) == 0
    for t in enc.tokens:
        c = t.lstrip("▁")[:1]
        assert not (c and (unicodedata.category(c) in ("Mn", "Mc") or c in "‌‍")), t


@pytest.mark.parametrize("s", ["क्ष", "क्‍ष", "क्‌ष", "జవహర్‌లాల్", "श‍र", "بھارَت", "क़"])
def test_joiners_and_normalisation(tok, s):
    enc = tok.encode(s)
    nfc = unicodedata.normalize("NFC", s)
    assert tok.decode(enc.ids) == nfc
    assert tok.decode(enc.ids).count("‌") == s.count("‌")
    assert tok.decode(enc.ids).count("‍") == s.count("‍")
    assert tok.encode(unicodedata.normalize("NFD", s)).ids == enc.ids
    assert "<unk>" not in enc.tokens


def test_zwj_and_zwnj_forms_tokenize_differently(tok):
    a, b, c = (tok.encode(x).tokens for x in ("क्ष", "क्‍ष", "क्‌ष"))
    assert a != b and a != c and b != c        # the joiners are kept, not silently dropped


# ------------------------------------------------------------------ website data
def test_site_artifacts_identical_to_release():
    assert (ROOT / "site" / "tokenizer.json").read_bytes() == TOK_PATH.read_bytes()
    rows = (ROOT / "site" / "tokens.tsv").read_text(encoding="utf-8").rstrip("\n").split("\n")
    assert rows[0].startswith("id\t")
    assert len(rows) == 10_001
    vocab = json.loads(TOK_PATH.read_text(encoding="utf-8"))["model"]["vocab"]
    inv = {i: t for t, i in vocab.items()}
    for r in rows[1:]:
        i, t = r.split("\t")[:2]
        assert json.loads(t) == inv[int(i)]


def test_site_numbers_match_verification():
    site = json.loads((ROOT / "site" / "data" / "site_data.json").read_text(encoding="utf-8"))
    summ = json.loads((ROOT / "verify" / "out" / "verification_summary.json").read_text(encoding="utf-8"))
    F = site["final"]
    assert F["tokenizer_sha256"] == summ["tokenizer_sha256"] == hashlib.sha256(TOK_PATH.read_bytes()).hexdigest()
    assert F["vocab_size"] == 10_000
    assert site["langs"] == summ["languages"]
    for l in LANGS:
        assert F["per_lang"][l]["tokens"] == summ["tokens"][l]
        assert F["per_lang"][l]["words"] == summ["words"][l]
        assert F["per_lang"][l]["fertility"] == summ["fertility"][l]
    assert F["spread"] == summ["spread"] and F["score"] == summ["score"]
    assert F["spread_exact"] == summ["spread_exact"]
    assert F["min_lang"] == summ["min_lang"] and F["max_lang"] == summ["max_lang"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_independent_js_implementation_agrees(tok):
    out = subprocess.run(["node", "verify/verify_js.mjs", str(TOK_PATH), "data/eval", ",".join(LANGS)],
                         cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    js = json.loads(out)
    for l in LANGS:
        assert js["languages"][l]["tokens"] == len(tok.encode(page(l)).ids)
        assert js["languages"][l]["grapheme_boundary_violations"] == 0
        assert js["languages"][l]["roundtrip_exact"]


# ------------------------------------------------------------------ reproducible training
@pytest.mark.slow
def test_retraining_reproduces_the_released_tokenizer():
    if not (ROOT / "data" / "raw" / "external").exists():
        pytest.skip("external training text not downloaded (run src/fetch_external.py)")
    from pipeline import train
    weights = {tuple(k.rsplit(".", 1)): v for k, v in FINAL["weights"].items()}
    t, _ = train(FINAL["config"], LANGS, weights)
    assert hashlib.sha256(t.to_str().encode("utf-8")).hexdigest() == \
        hashlib.sha256(Tokenizer.from_file(str(TOK_PATH)).to_str().encode("utf-8")).hexdigest()


def test_published_eval_texts_identical_and_hashes_match_sha256sum():
    man = json.loads((ROOT / "site" / "data" / "eval" / "manifest.json").read_text(encoding="utf-8"))
    assert man == MANIFEST
    for l in LANGS:
        pub = (ROOT / "site" / "data" / "eval" / f"{l}.txt").read_bytes()
        assert pub == (ROOT / "data" / "eval" / f"{l}.txt").read_bytes()
        assert hashlib.sha256(pub).hexdigest() == MANIFEST["pages"][l]["file_sha256"]


def test_tradeoff_starts_at_released_tokenizer():
    X = json.loads((ROOT / "experiments" / "site_experiments.json").read_text(encoding="utf-8"))
    summ = json.loads((ROOT / "verify" / "out" / "verification_summary.json").read_text(encoding="utf-8"))
    p0 = X["tradeoff"]["points"][0]
    assert p0["en_weight"] == 1.0 and abs(p0["spread"] - summ["spread"]) < 1e-12
