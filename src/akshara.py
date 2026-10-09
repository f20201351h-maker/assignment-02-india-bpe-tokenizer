"""Akshara (orthographic syllable) segmentation, used for the Indic-safety rule and checks.

A token boundary is "safe" only between aksharas. Rule, applied to every script:
  * a combining mark (Unicode category Mn/Mc/Me: matras, virama/halant, nukta,
    anusvara, candrabindu, visarga, ...) attaches to the preceding character;
  * ZWJ (U+200D) and ZWNJ (U+200C) attach to the preceding character;
  * a consonant that directly follows a virama (optionally with a ZWJ in between)
    joins the same akshara -> conjuncts such as क्ष, త్ర, ಕ್ಷ stay whole.
    After virama + ZWNJ the next consonant starts a new akshara (ZWNJ explicitly
    asks for the consonants to stay visually separate);
  * Tamil is the exception: the pulli does not form conjuncts, so a consonant after
    a pulli starts a new akshara.
Everything else (letters, independent vowels, digits, punctuation) starts a new unit.
"""
import functools
import unicodedata

ZWJ, ZWNJ = "\u200d", "\u200c"
NO_CONJUNCT_SCRIPTS = {"TAMIL"}


@functools.lru_cache(maxsize=None)
def _script(ch):
    name = unicodedata.name(ch, "")
    return name.split(" ")[0] if name else ""


@functools.lru_cache(maxsize=None)
def _is_mark(ch):
    return unicodedata.category(ch) in ("Mn", "Mc", "Me")


@functools.lru_cache(maxsize=None)
def _is_virama(ch):
    return unicodedata.combining(ch) == 9


@functools.lru_cache(maxsize=2_000_000)
def _aksharas(word: str):
    units = []
    cur = ""
    for ch in word:
        if not cur:
            cur = ch
            continue
        if _is_mark(ch) or ch in (ZWJ, ZWNJ):
            cur += ch
            continue
        # conjunct: ... virama [ZWJ] + consonant of the same script
        tail = cur[:-1] if cur.endswith(ZWJ) else cur
        if (tail and _is_virama(tail[-1]) and unicodedata.category(ch) == "Lo"
                and _script(ch) == _script(tail[-1]) and _script(ch) not in NO_CONJUNCT_SCRIPTS):
            cur += ch
            continue
        units.append(cur)
        cur = ch
    if cur:
        units.append(cur)
    return tuple(units)


def aksharas(word: str):
    return list(_aksharas(word))


def boundaries(word: str):
    """Character offsets (0 < i < len) where a token boundary is allowed."""
    out, pos = set(), 0
    for u in aksharas(word)[:-1]:
        pos += len(u)
        out.add(pos)
    return out


def bad_splits(word: str, pieces):
    """Number of internal piece boundaries that fall inside an akshara."""
    allowed = boundaries(word)
    bad, pos = 0, 0
    for p in pieces[:-1]:
        pos += len(p)
        if pos not in allowed:
            bad += 1
    return bad
