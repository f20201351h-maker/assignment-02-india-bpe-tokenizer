"""Run both independent verifiers in fresh processes and cross-check them.

    python verify/run_verification.py [tokenizer.json] [langs]

Writes verify/out/{python_report,js_report,unicode_checks,verification_summary}.json.
Nothing here imports src/; the only inputs are the exported tokenizer.json and data/eval/.
"""
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verify" / "out"
TOK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tokenizer" / "tokenizer.json"
LANGS = sys.argv[2] if len(sys.argv) > 2 else "en,hi,te,ur"

UNICODE_CASES = [
    ("Devanagari conjunct क्ष", "क्ष"),
    ("क्ष with ZWJ (half form)", "क्‍ष"),
    ("क्ष with ZWNJ (explicit halant)", "क्‌ष"),
    ("Hindi word with conjunct", "शक्ति"),
    ("Telugu name with ZWNJ", "జవహర్‌లాల్ నెహ్రూ"),
    ("Telugu conjunct cluster", "స్వాతంత్ర్యం"),
    ("Telugu anusvara word", "భారతదేశం"),
    ("Hindi nukta letter (precomposed ज़, NFC decomposes it)", "ज़रा"),
    ("Urdu with diacritic", "بھارَت"),
    ("Mixed scripts", "India भारत భారత بھارت 1947"),
]


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}\n{p.stderr}")
    return p


def unicode_checks(tok_path, langs):
    tok = Tokenizer.from_file(str(tok_path))
    rows = []
    for name, s in UNICODE_CASES:
        enc = tok.encode(s)
        dec = tok.decode(enc.ids)
        expect = unicodedata.normalize("NFC", s)
        rows.append({
            "case": name, "input": s, "tokens": enc.tokens,
            "roundtrip": dec == expect,
            "joiners_preserved": all(dec.count(j) == expect.count(j) for j in ("‌", "‍")),
            "nfd_same_tokens": tok.encode(unicodedata.normalize("NFD", s)).ids == enc.ids,
            "unk": enc.tokens.count("<unk>"),
        })
    # no token on an evaluation page may start with a combining mark or a joiner
    bad_starts = {}
    for l in langs:
        text = (ROOT / "data" / "eval" / f"{l}.txt").read_text(encoding="utf-8")
        n = 0
        for t in tok.encode(text).tokens:
            c = t.lstrip("▁")[:1]
            if c and (unicodedata.category(c) in ("Mn", "Mc", "Me") or c in "‌‍"):
                n += 1
        bad_starts[l] = n
    return {"cases": rows, "tokens_starting_with_mark_or_joiner": bad_starts}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "verify/verify_python.py", str(TOK), "data/eval", LANGS, str(OUT / "python_report.json")])
    run(["node", "verify/verify_js.mjs", str(TOK), "data/eval", LANGS, str(OUT / "js_report.json")])
    py = json.loads((OUT / "python_report.json").read_text(encoding="utf-8"))
    js = json.loads((OUT / "js_report.json").read_text(encoding="utf-8"))
    uc = unicode_checks(TOK, LANGS.split(","))
    (OUT / "unicode_checks.json").write_text(json.dumps(uc, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    langs = LANGS.split(",")
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    check("vocabulary size is exactly 10,000 (Python)", py["vocab_size_hf"] == 10000, str(py["vocab_size_hf"]))
    check("vocabulary size is exactly 10,000 (JS)", js["vocab_size"] == 10000, str(js["vocab_size"]))
    check("no added tokens outside the BPE vocabulary", py["added_tokens"] == ["<unk>"] or py["added_tokens"] == [],
          str(py["added_tokens"]))
    check("token ids are 0..9999 with no gaps", py["vocab_ids_contiguous"] and js["vocab_ids_contiguous"])
    check("every merge refers to tokens in the vocabulary", py["merges_valid"] and js["merges_valid"])
    check("model type is BPE", py["model_type"] == "BPE")
    check("no post-processor adds extra tokens", py["post_processor"] is None)
    check("both verifiers read the same tokenizer file", py["tokenizer_sha256"] == js["tokenizer_sha256"], py["tokenizer_sha256"])
    for l in langs:
        p, j = py["languages"][l], js["languages"][l]
        check(f"{l}: evaluation text matches pinned hash", p["sha_matches_manifest"] and j["sha_matches_manifest"])
        check(f"{l}: evaluation file bytes match pinned file hash (sha256sum)", p["file_sha_matches_manifest"])
        check(f"{l}: same token count with the final newline kept, and line by line",
              p["tokens"] == p["tokens_untrimmed_file"] == p["tokens_line_by_line"],
              f"{p['tokens']} / {p['tokens_untrimmed_file']} / {p['tokens_line_by_line']}")
        check(f"{l}: word count agrees (split / regex / JS)", p["words"] == p["words_regex_check"] == j["words"],
              f"{p['words']} / {p['words_regex_check']} / {j['words']}")
        check(f"{l}: token count agrees (HF Python / independent JS)", p["tokens"] == j["tokens"],
              f"{p['tokens']} / {j['tokens']}")
        check(f"{l}: no <unk> tokens", p["unk_tokens"] == 0 and j["unk_tokens"] == 0)
        check(f"{l}: decode(encode(text)) is lossless", p["roundtrip_exact"] and j["roundtrip_exact"])
        check(f"{l}: NFD input gives identical tokens", p["nfd_input_same_tokens"])
        check(f"{l}: deterministic encoding", p["deterministic"])
        check(f"{l}: no token boundary inside a grapheme cluster (ICU)", j["grapheme_boundary_violations"] == 0,
              str(j["grapheme_boundary_violations"]))
        check(f"{l}: no token starts with a combining mark or joiner",
              uc["tokens_starting_with_mark_or_joiner"][l] == 0, str(uc["tokens_starting_with_mark_or_joiner"][l]))
    check("min/max languages agree", py["min_lang"] == js["min"]["lang"] and py["max_lang"] == js["max"]["lang"])
    check("spread agrees (exact fraction vs JS float)", abs(py["spread"] - js["spread"]) < 1e-15,
          f"{py['spread_exact']} = {py['spread']!r}")
    check("score agrees", abs(py["score"] - js["score"]) / py["score"] < 1e-12, f"{py['score']!r}")
    check("Unicode test strings round-trip with joiners preserved",
          all(r["roundtrip"] and r["joiners_preserved"] and r["nfd_same_tokens"] and r["unk"] == 0 for r in uc["cases"]))
    summary = {
        "tokenizer": TOK.relative_to(ROOT).as_posix() if TOK.is_relative_to(ROOT) else TOK.as_posix(),
        "tokenizer_sha256": py["tokenizer_sha256"], "languages": langs,
        "fertility": {l: py["languages"][l]["fertility"] for l in langs},
        "fertility_exact": {l: py["languages"][l]["fertility_exact"] for l in langs},
        "tokens": {l: py["languages"][l]["tokens"] for l in langs},
        "words": {l: py["languages"][l]["words"] for l in langs},
        "min_lang": py["min_lang"], "max_lang": py["max_lang"],
        "spread": py["spread"], "spread_exact": py["spread_exact"],
        "score": py["score"], "score_exact": py["score_exact"], "score_decimal": py["score_decimal_30"],
        "all_pass": all(c["pass"] for c in checks), "checks": checks,
        "versions": {"python": py["python"], "tokenizers": py["tokenizers"], "python_unicode": py["unicode_version"],
                     "node": js["node"], "node_icu_unicode": js["icu_unicode"]},
    }
    (OUT / "verification_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] else ""))
    print(f"\nspread = {py['spread_exact']} = {py['spread']!r}\nscore  = {py['score_decimal_30']}")
    print("ALL CHECKS PASS" if summary["all_pass"] else "SOME CHECKS FAILED")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
