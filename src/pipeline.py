"""Training + evaluation pipeline for the 10K multilingual BPE tokenizer.

Overview
--------
1. Text from each (language, source) is normalised and pre-tokenised with exactly the
   normaliser / pre-tokeniser that the final tokenizer.json uses, giving word counts.
2. Each (language, source) gets a weight. Weighted word counts are fed to the Hugging Face
   BPE trainer (a weight of 2 means "as if this text appeared twice").
3. Optional akshara-first stage: every frequent multi-character akshara is first glued into
   one unit by fixed merges placed at the top of the merge list; the BPE trainer then only
   learns merges between whole aksharas.
4. The final tokenizer is a standard Hugging Face BPE tokenizer.json with exactly
   `vocab_size` entries (special tokens + base characters + merges).
"""
import collections
import gzip
import hashlib
import json
import math
import pathlib
import pickle
import time

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from akshara import aksharas, bad_splits
from textnorm import clean_text, count_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
EXT_DIR = ROOT / "data" / "raw" / "external"
CACHE = ROOT / "data" / "cache"
SPACE = "▁"  # ▁
PUA0 = 0xF0000     # Supplementary Private Use Area-A, used only inside the trainer
PUNCT_RE = r"▁?[[\p{P}\p{S}]&&[^▁]]+"

DEFAULTS = dict(
    vocab_size=10_000,
    norm="nfc",            # nfc | nfkc
    pretok="ms",           # ms (SentencePiece-style metaspace) | ms_punct (punctuation split off)
    akshara=False,         # akshara-first merges
    akshara_min=0.0,       # weighted-frequency threshold for an akshara to be glued (eval aksharas always are)
    byte_fallback=False,
    alpha_min=0.0,         # chars rarer than this (weighted) are dropped from training (eval chars always kept)
    ext_words=600_000,     # words of additional Wikipedia text per language
    min_frequency=0,
)


# ----------------------------------------------------------------------------- pipeline parts
def build_normalizer(norm):
    first = normalizers.NFC() if norm == "nfc" else normalizers.NFKC()
    return normalizers.Sequence([first, normalizers.Replace(Regex(r"\s+"), " "), normalizers.Strip()])


def build_pretokenizer(pretok):
    ms = pre_tokenizers.Metaspace(replacement=SPACE, prepend_scheme="always", split=True)
    if pretok == "ms":
        return ms
    if pretok == "ms_punct":
        return pre_tokenizers.Sequence([ms, pre_tokenizers.Split(Regex(PUNCT_RE), behavior="isolated")])
    raise ValueError(pretok)


def build_decoder(byte_fallback):
    ms = decoders.Metaspace(replacement=SPACE, prepend_scheme="always", split=True)
    if byte_fallback:
        # (Fuse before Metaspace drops the spaces; this Llama-style chain decodes correctly)
        return decoders.Sequence([decoders.Replace(SPACE, " "), decoders.ByteFallback(), decoders.Fuse(),
                                  decoders.Strip(" ", 1, 0)])
    return ms


# ----------------------------------------------------------------------------- data
def eval_text(lang):
    return (EVAL_DIR / f"{lang}.txt").read_text(encoding="utf-8").rstrip("\n")


def ext_docs(lang, split, max_words=None):
    out, n = [], 0
    with gzip.open(EXT_DIR / f"{lang}.{split}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if max_words is not None and n >= max_words:
                break
            t = clean_text(json.loads(line)["text"])
            out.append(t)
            n += count_words(t)
    return out


def word_counts(lang, source, norm, pretok, ext_words=None):
    """Counter of pre-tokens (e.g. '▁India,') for one language/source, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{lang}.{source}.{norm}.{pretok}.{ext_words}"
    path = CACHE / f"wc.{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    nz, pt = build_normalizer(norm), build_pretokenizer(pretok)
    docs = [eval_text(lang)] if source == "eval" else ext_docs(lang, source, ext_words)
    c = collections.Counter()
    for d in docs:
        for line in d.split("\n"):
            c.update(p for p, _ in pt.pre_tokenize_str(nz.normalize_str(line)))
    path.write_bytes(pickle.dumps(c))
    return c


# ----------------------------------------------------------------------------- akshara stage
def _bpe_apply(symbols, ranks):
    """Plain rank-ordered BPE on a list of symbols (same rule as the HF BPE model)."""
    symbols = list(symbols)
    while len(symbols) > 1:
        best, bi = None, -1
        for i in range(len(symbols) - 1):
            r = ranks.get((symbols[i], symbols[i + 1]))
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            break
        symbols[bi:bi + 2] = [symbols[bi] + symbols[bi + 1]]
    return symbols


def akshara_merges(akshara_list):
    """Fixed merges that assemble each akshara (most frequent first) into one symbol.

    For every akshara we replay the merges found so far; while it is still in several
    pieces we add a merge of its first two pieces. Because these merges come before all
    learned merges, encoding any word first assembles all of its listed aksharas.
    """
    merges, ranks = [], {}
    for a in akshara_list:
        while True:
            parts = _bpe_apply(list(a), ranks)
            if len(parts) == 1:
                break
            pair = (parts[0], parts[1])
            ranks[pair] = len(merges)
            merges.append(pair)
    return merges


# ----------------------------------------------------------------------------- training
def train(cfg, langs, weights, log=None):
    """weights: {(lang, source): float}; source in {'eval', 'train'} (external train pool)."""
    c = {**DEFAULTS, **cfg}
    t0 = time.time()
    # 1. weighted word counts
    wc = collections.Counter()
    for (lang, src), w in weights.items():
        if w <= 0:
            continue
        cnt = word_counts(lang, src, c["norm"], c["pretok"], c["ext_words"] if src == "train" else None)
        for word, n in cnt.items():
            wc[word] += n * w
    eval_words = collections.Counter()
    for lang in langs:
        if weights.get((lang, "eval"), 0) > 0:
            eval_words.update(word_counts(lang, "eval", c["norm"], c["pretok"]))
    # 2. alphabet: characters in the (weighted) data; rare ones dropped unless seen on an eval page
    char_w = collections.Counter()
    for word, n in wc.items():
        for ch in word:
            char_w[ch] += n
    keep_chars = {ch for ch, n in char_w.items() if n >= c["alpha_min"]} | {ch for w in eval_words for ch in w}
    wc = collections.Counter({w: n for w, n in wc.items() if all(ch in keep_chars for ch in w)})

    # 3. akshara-first: choose aksharas to glue, map them to private-use symbols for the trainer
    stage1, pua_of, ak_of = [], {}, {}
    if c["akshara"]:
        ak_w = collections.Counter()
        for word, n in wc.items():
            for a in aksharas(word):
                if len(a) > 1:
                    ak_w[a] += n
        forced = {a for w in eval_words for a in aksharas(w) if len(a) > 1}
        chosen = [a for a, n in ak_w.most_common() if n >= c["akshara_min"] or a in forced]
        for i, a in enumerate(chosen):
            pua_of[a] = chr(PUA0 + i)
            ak_of[chr(PUA0 + i)] = a
        stage1 = akshara_merges(chosen)

    def map_word(word):
        if not c["akshara"]:
            return word
        out = []
        for a in aksharas(word):
            if len(a) == 1:
                out.append(a)
            elif a in pua_of:
                out.append(pua_of[a])
            else:  # rare akshara: isolate its characters so no learned merge touches it
                out.append(" " + " ".join(a) + " ")
        return "".join(out)

    mapped = collections.Counter()
    for word, n in wc.items():
        mapped[map_word(word)] += n
    # 4. base vocabulary (real characters) and stage-1 tokens
    specials = ["<unk>"]
    byte_tokens = [f"<0x{i:02X}>" for i in range(256)] if c["byte_fallback"] else []
    base_chars = set()
    for word in mapped:
        for ch in word:
            if ch == " ":
                continue
            base_chars.update(ak_of.get(ch, ch))
    base_chars = sorted(base_chars)
    stage1_tokens = []
    seen = set(base_chars)
    for a, b in stage1:
        if a + b not in seen:
            seen.add(a + b)
            stage1_tokens.append(a + b)
    base_size = len(specials) + len(byte_tokens) + len(base_chars) + len(stage1_tokens)
    n_merges = c["vocab_size"] - base_size
    if n_merges <= 0:
        raise ValueError("vocabulary budget exhausted by base symbols")
    # 5. feed weighted counts to the HF trainer (integer counts; scale so rounding is negligible)
    total = sum(mapped.values())
    scale = c.get("mass", 6_000_000) / total
    alphabet2 = sorted({ch for w in mapped for ch in w if ch != " "})
    trainer_tok = Tokenizer(models.BPE())
    trainer_tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    slack = 64  # a few spare merges in case an unmapped merge duplicates an existing token
    trainer = trainers.BpeTrainer(vocab_size=len(alphabet2) + n_merges + slack, min_frequency=c["min_frequency"],
                                  show_progress=False, initial_alphabet=alphabet2, special_tokens=[])

    def feed():
        for word, n in mapped.items():
            # unbiased deterministic rounding: floor(x + u), u = hash(word) in [0, 1)
            u = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16) / 2**32
            k = int(math.floor(n * scale + u))
            while k > 0:
                m = min(k, 2000)
                yield " ".join([word] * m)
                k -= m

    trainer_tok.train_from_iterator(feed(), trainer)
    raw = json.loads(trainer_tok.to_str())["model"]
    merges2 = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in raw["merges"]]

    def unmap(s):
        return "".join(ak_of.get(ch, ch) for ch in s)

    merges2 = [(unmap(a), unmap(b)) for a, b in merges2]
    # 6. assemble vocabulary: id order = specials, bytes, characters, akshara merges, learned merges
    vocab = {}
    for t in specials + byte_tokens + base_chars + stage1_tokens:
        vocab.setdefault(t, len(vocab))
    merges = list(stage1)
    for a, b in merges2:
        if len(vocab) >= c["vocab_size"]:
            break
        if a + b in vocab:
            continue
        vocab[a + b] = len(vocab)
        merges.append((a, b))
    if len(vocab) != c["vocab_size"]:
        raise RuntimeError(f"vocab size {len(vocab)} != {c['vocab_size']} (corpus too small for the budget?)")
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>",
                               byte_fallback=c["byte_fallback"], fuse_unk=False))
    tok.normalizer = build_normalizer(c["norm"])
    tok.pre_tokenizer = build_pretokenizer(c["pretok"])
    tok.decoder = build_decoder(c["byte_fallback"])
    info = dict(n_base_chars=len(base_chars), n_stage1=len(stage1_tokens), n_learned=len(merges) - len(stage1),
                n_specials=len(specials) + len(byte_tokens), train_seconds=round(time.time() - t0, 1),
                weighted_mass={f"{k[0]}.{k[1]}": v for k, v in weights.items()})
    return tok, info


# ----------------------------------------------------------------------------- evaluation
def fertility(tok, text):
    return len(tok.encode(text).ids), count_words(text)


def akshara_violations(tok, text):
    """Token boundaries inside an akshara, counted per pre-token."""
    enc = tok.encode(text)
    groups = collections.defaultdict(list)
    for t, wid in zip(enc.tokens, enc.word_ids):
        groups[wid].append(t)
    bad = 0
    for pieces in groups.values():
        if any(p.startswith("<0x") for p in pieces):
            bad += 1
            continue
        bad += bad_splits("".join(pieces), pieces)
    return bad, len(enc.ids)


_heldout_cache = {}


def heldout_text(lang, max_words=60_000):
    if lang not in _heldout_cache:
        _heldout_cache[lang] = "\n".join(ext_docs(lang, "heldout", max_words))
    return _heldout_cache[lang]


def evaluate(tok, langs, heldout=True, akshara_check=True):
    res = {"langs": langs, "eval": {}, "heldout": {}}
    for lang in langs:
        text = eval_text(lang)
        n_tok, n_words = fertility(tok, text)
        r = {"tokens": n_tok, "words": n_words, "fertility": n_tok / n_words,
             "unk": tok.encode(text).ids.count(tok.token_to_id("<unk>"))}
        if akshara_check:
            r["akshara_violations"] = akshara_violations(tok, text)[0]
        res["eval"][lang] = r
        if heldout:
            ht = heldout_text(lang)
            ht_tok, ht_words = fertility(tok, ht)
            h = {"tokens": ht_tok, "words": ht_words, "fertility": ht_tok / ht_words,
                 "unk": tok.encode(ht).ids.count(tok.token_to_id("<unk>"))}
            if akshara_check:
                h["akshara_violations"] = akshara_violations(tok, ht)[0]
            res["heldout"][lang] = h
    f = {l: res["eval"][l]["fertility"] for l in langs}
    hi_l, lo_l = max(f, key=f.get), min(f, key=f.get)
    spread = f[hi_l] - f[lo_l]
    res.update(max_lang=hi_l, min_lang=lo_l, max_fertility=f[hi_l], min_fertility=f[lo_l], spread=spread,
               score=(1000 / spread) if spread > 0 else math.inf, vocab_size=tok.get_vocab_size())
    if heldout:
        hf = [res["heldout"][l]["fertility"] for l in langs]
        res["heldout_spread"] = max(hf) - min(hf)
        res["heldout_max"] = max(hf)
    return res


def tokenizer_sha(tok):
    return hashlib.sha256(tok.to_str().encode("utf-8")).hexdigest()
