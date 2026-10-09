"""Text cleaning and the word-count rule. Fixed BEFORE any optimisation.

Cleaning (identical for every language):
  1. Unicode NFC normalisation.
  2. Every whitespace character (str.isspace: tabs, NBSP, thin space, ...) becomes a plain
     space, except line breaks, which separate paragraphs.
  3. Runs of spaces collapse to one space; each line is stripped; empty lines are dropped.
Nothing else is removed. Zero-width joiner / non-joiner (U+200D / U+200C) and all other
characters are kept exactly as Wikipedia serves them.

Word count (the fertility denominator), as used in the course's worked example
("Artificial intelligence is changing the world" = 6 words): the number of
whitespace-separated chunks of the cleaned text, i.e. Python `len(text.split())`.
Punctuation attached to a word stays part of that word; a stand-alone dash between
two spaces counts as one word.
"""
import unicodedata

LINE_BREAKS = "\n\r  \x0b\x0c\x1c\x1d\x1e\x85"


def clean_text(raw: str) -> str:
    t = unicodedata.normalize("NFC", raw)
    t = "".join("\n" if ch in LINE_BREAKS else (" " if ch.isspace() else ch) for ch in t)
    lines = (" ".join(line.split(" ")).strip() for line in t.split("\n"))
    lines = (" ".join(w for w in line.split(" ") if w) for line in lines)
    return "\n".join(line for line in lines if line)


def count_words(text: str) -> int:
    return len(text.split())
