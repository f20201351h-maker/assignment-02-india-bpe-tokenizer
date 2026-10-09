"""Explanatory text for the website. Numbers are filled in by build_site.py from the data files."""

METHOD = """
<h2>Method</h2>

<h3>BPE in one paragraph</h3>
<p>Byte-Pair Encoding starts from single characters. It counts every pair of neighbouring symbols in the
training text, merges the most frequent pair into a new token, writes that merge down, and repeats until the
vocabulary is full. To tokenize new text the same merges are replayed in the same order. The downloadable
<code>tokenizer.json</code> contains exactly that: {vocab} vocabulary entries and the ordered list of
{n_merges:,} merges (plus the normaliser and pre-tokenizer settings), so anyone can load it with
<code>Tokenizer.from_file("tokenizer.json")</code> (Hugging Face <code>tokenizers</code> 0.20 or newer; I
checked 0.20.3, 0.21.0 and 0.23.1 and they give identical token counts).</p>

<h3>Fertility and the score</h3>
<p>Fertility is the number of tokens needed per word, measured on one page:</p>
<span class="formula">fertility = tokens on the page / words on the page</span>
<p>Words are the whitespace-separated chunks of the cleaned page text, as in the standard worked example
("Artificial intelligence is changing the world" = 6 words). Punctuation attached to a word is part of that
word. The word counting rule was fixed before any tuning and never changed. The score is</p>
<span class="formula">score = 1000 / (largest fertility − smallest fertility)</span>
<p>computed from the exact token and word counts (the gap is an exact fraction, shown in the result card).</p>

<h3>Getting the pages</h3>
<p>All four pages were downloaded through the official MediaWiki API as plain text (the TextExtracts
extension with <code>explaintext=1&amp;exsectionformat=plain</code>, so section headings are plain lines).
Wikimedia removes tables, infoboxes, figures, reference lists and navigation boxes on the server, in the same
way for every language, so I did not write any language-specific cleaning. My own cleaning is only: Unicode
NFC, every kind of whitespace (tabs, non-breaking spaces, thin spaces) turned into a normal space, and empty
lines dropped. The cleaned texts are frozen together with their revision ids and can be downloaded at the top
of this page, so the score can always be recomputed on exactly the same text.</p>

<h3>Spaces and punctuation</h3>
<p>As in SentencePiece, a space becomes the visible symbol "▁" attached to the following word, so decoding
gives back the (normalised) text: every run of whitespace, including line breaks, comes back as one space.
I first tried also splitting punctuation into separate pieces, but then every comma and full stop is a token of
its own; on the English page that alone makes {punct_floor:.3f} the lowest fertility possible, which is above
the 1.2 target. So punctuation is allowed to merge into words (for example "▁India," can be one token).
Tokens never cross a space, so a token is always inside one word and fertility can never go below 1.</p>

<h3>Characters and the unknown token</h3>
<p>The base vocabulary has {n_chars} single characters: every character that occurs on the four pages plus the
ones that are common in the extra training text. As with word-level tokenizers, anything
outside this set becomes <code>&lt;unk&gt;</code>; here that only happens for single unseen characters
(for example most Telugu and Urdu digits, emoji or other scripts), never for a whole word. There is no
<code>&lt;unk&gt;</code> on the four pages. On {heldout_words:,} words of unseen Wikipedia text it happened
{heldout_unk:,} times. I tried byte fallback (256 byte tokens, so nothing is ever unknown), but it cost about
{bytefb_cost:.2f} in fertility on the pages, so I did not use it.</p>

<h3>Indic text: akshara-first merges</h3>
<p>An akshara (syllable) such as క్తి or क्ष is several Unicode characters: consonant, virama (halant),
consonant, vowel sign. Plain BPE does not know this and happily puts a token boundary in the middle of one,
e.g. a token that ends with a halant or starts with a matra. My first plain-BPE tokenizer (pages only, equal
weights) did this {base_akv:,} times on the four pages, and plain BPE with my final training recipe still did it
{plain_akv:,} times. So the first merges in my tokenizer glue every akshara together:
vowel signs, anusvara, nukta and virama stay with their consonant, consonants joined by a virama stay
together (conjuncts), a ZWJ keeps the conjunct together, and a ZWNJ is respected as an explicit request to
keep the consonants apart. Only after that does BPE learn merges, and those merges only join whole aksharas.
Result: no token boundary falls inside an akshara anywhere on the four evaluation pages. I checked this with my
own rule and, independently, with the browser's Unicode grapheme segmenter (Intl.Segmenter). This costs about
{akshara_cost:.2f} in fertility compared with plain BPE (see the recipe table). Characters are never split into
bytes, so a single Unicode character is never cut either. The guarantee is for the four pages: an akshara that
never appeared there (for example क्‍ष written with a ZWJ) is spelled out character by character, with the
joiner kept as its own token. On the unseen Wikipedia text this happens {heldout_akv:,} times, compared with
{plain_heldout_akv:,} cuts for plain BPE.</p>

<h3>Training data</h3>
<p>Any data may be used to build the tokenizer, as long as fertility is measured on
the four India pages. My training text is therefore the four evaluation pages plus about {ext_words:,} words of
other Wikipedia articles per language (the 2023 <code>wikimedia/wikipedia</code> dump; the India article itself
excluded). Inside each language, the page counts {eval_w:g} times as much as the same amount of general text;
after balancing the language weights, the pages count {page_weights}. Because the evaluation pages are part of
the training data, fertility on them is lower than on unseen text; the language table therefore also shows
held-out fertility on Wikipedia articles that were not used for training. The extra articles matter for a
practical reason: almost every word on these pages occurs only once, so with the pages alone BPE spends its
last merges in big blocks on one language at a time and the fertilities jump around: over my first balancing
rounds Telugu went 2.01 → 1.00 → 2.04 → 1.00 → 1.60 (1.00 means every Telugu word had become a single token).
The extra text acts as a tie-breaker (generally common words first) and makes the result change smoothly with
the weights.</p>

<h3>Fourth language</h3>
<p>I did not pick the fourth language by hand. I ran the same recipe for {n_fourth} candidates, balanced the
weights for each, and compared the fertility level at which all four languages meet (the worst of the four,
lower is better). Urdu gave the lowest level ({ur_level:.3f}), ahead of Punjabi ({pa_level:.3f}) and Marathi
({mr_level:.3f}). Urdu is the language closest to Hindi in speech but written in a different (Perso-Arabic)
script, and its India article ({ur_words:,} words) is longer than the Telugu one ({te_words:,} words). Even
candidates with much smaller pages (Gujarati, Kannada, Sanskrit) came out worse, so the choice is not explained
by page size.</p>

<h3>Optimising the score</h3>
<p>The score only depends on the gap between the best and the worst language. Each language gets a sampling
weight in the training text. After each training run, languages above the average fertility get more weight
and those below get less, and the tokenizer is retrained from scratch. Once the four values were close, a
small random search around the best weights shrank the gap further. Giving more weight to the weakest
language (Telugu has {te_mult:.1f}× the weight of English) is what closes the gap. No language was made
worse on purpose: among the {n_recipe_runs} runs with the final recipe, none had a lower worst-language fertility
than the released tokenizer ({final_max:.6f}). In total I ran {n_runs:,} training runs ({n_distinct:,} distinct
tokenizers; the rest were deliberate re-runs of the same settings, for example to check that training is
deterministic).</p>
"""

CAVEAT = (
    "<b>Honest caveats.</b> "
    "<b>(1) English is not at 1.2.</b> English fertility is {en:.3f}, above the 1.2 target set for the task "
    "(it is within the looser 1.6 bound). English ≤ 1.2 is reachable (giving English 6× its "
    "weight gives {reach_en:.3f}), but the 10,000 tokens are shared, so Telugu then needs {reach_worst:.2f} tokens "
    "per word and the score falls to {reach_score:,.0f}. With tokens that never cross a space I could not get all "
    "four languages to 1.2 together: English, Hindi and Telugu alone only meet at {lvl3:.3f}. See "
    "<a href=\"#experiments\">Why English is not at 1.2</a>. "
    "<b>(2) The evaluation pages are in the training data</b> (the task allows any training data); on unseen Wikipedia text the "
    "fertilities are {heldout_range}. "
    "<b>(3) The score belongs to the exact page texts.</b> {live_sentence} Extracted differently, the same "
    "pages give other fertilities and a different score: about {wiki_score:,.0f} with the "
    "'== Heading ==' format of the <code>wikipedia</code> Python package, and about {alt_lo:,.0f}–{alt_hi:,.0f} when "
    "the text is taken from the rendered HTML instead (which adds lists, references and captions). English is only {en_margin:.1f} token above the best language, so even one token more "
    "or less on a page changes the score a lot. Download the texts above to check the number."
)

INTRO = ("{n:,} training runs ({d:,} distinct tokenizers) in total. Every run is stored in "
         "<code>experiments/</code> (configuration, weights, fertilities, checks and the tokenizer hash).")

BASELINE_NOTE = ("Baseline: plain BPE trained only on the four pages with equal weights (no akshara rule, no "
                 "balancing). Telugu was far behind and the score was {score:,.0f}. The final tokenizer brings all "
                 "four languages to the same fertility.")

FOURTH_NOTE = ("Same recipe for every candidate, weights balanced automatically. Each dot is the fertility at "
               "which all four languages meet (the worst of the four). Hollow dots are pages under {minw:,} "
               "words, shown for comparison only. Hover for page size and held-out fertility.")

TRADEOFF_NOTE = ("Starting from the released weights (×1 is the released tokenizer), only the English weight "
                 "was increased and the tokenizer retrained each time. English gets cheaper, but the 10,000 tokens "
                 "are shared, so the other languages get more expensive and the gap grows. English reaches 1.2 "
                 "between ×4 and ×6 (1.189 at ×6), where the score is {reach_score:,.0f} instead of {final_score:,.0f}.")

ABLATION_NOTE = ("Each row changes one thing in the recipe and re-balances the weights. 'Balanced fertility' is "
                 "the worst of the four languages at the best balance reached (lower is better), 'held-out' is "
                 "the worst language on unseen Wikipedia articles.")

TRACE_NOTE = ("Spread of each run in the final balancing and fine-tuning stage (log scale). Each point is a "
              "complete retrain of the tokenizer with slightly different language weights; the best one was "
              "kept.")

SENS_NOTE = ("The same tokenizer.json measured on other versions of the same pages. The reported score uses "
             "the pinned snapshots. Today's live pages give the same result in the same format, but extracting "
             "the text differently changes the fertilities by up to {alt_delta:.2f} and the score by one to three "
             "orders of magnitude, so the score is only meaningful together with the exact text it was measured on "
             "(downloadable at the top of the page).")
