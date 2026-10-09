# Research log — India BPE tokenizer

Decisions in the order they were made. Numbers quoted here are copied from the ledgers
(`experiments/results.jsonl` for stage A, `experiments/ledger/*.jsonl` afterwards); the
summary tables in `experiments/SUMMARY.md` are generated from those files by
`src/summarize.py`.

## 0. Fixed before any optimisation

* **Fertility** = tokens on the evaluation page / words on the evaluation page
  (lecture: "Artificial intelligence is changing the world" = 6 words, 8 tokens → 1.33).
  The denominator is running words, not unique words ("if your document had 1,000 words, you
  cannot represent those thousand words in more than 1,600 tokens").
* **Word count** = `len(text.split())` on the cleaned page text. Not changed afterwards.
* **Cleaning** = TextExtracts plain text (Wikimedia strips tables/infobox/references
  server-side, same rule for every language) → NFC → whitespace runs to one space, lines kept.
* **Score** = 1000 / (max fertility − min fertility), computed from exact token and word counts.
* Evaluation pages: en/hi/te + candidates, revision ids frozen in `data/eval/manifest.json`.
* Extra training text: `wikimedia/wikipedia` 20231101 dump, first ~600K words per language
  (articles split train/held-out by title hash; the India article itself is excluded).

## A. First baselines (en, hi, te, mr; plain BPE; 10,000 tokens)

* E0001 eval pages only, equal weights: en 1.417, hi 1.262, te 2.014, mr 1.538 → score 1,330.
* External text only: en 2.051, hi 1.599, te 2.299, mr 1.978 → score 1,430.
* Telugu is the bottleneck in every case.
* Plain BPE cuts inside aksharas very often (E0001: 653 / 1,038 / 972 bad boundaries on the
  hi / te / mr pages; ICU grapheme segmentation in Node agrees with these counts exactly).
* Splitting punctuation into separate pre-tokens makes 1.227 the *lowest possible* English
  fertility on this page (every comma/full stop becomes a token), above the 1.2 target, so the
  SentencePiece-style pre-tokenizer (space → "▁", punctuation may merge into words) is used.
* Eval-only training balances badly: almost every word on these pages occurs once, so all of
  a language's merges move together. Over the first balancing rounds Telugu went
  2.01 → 1.00 → 2.04 → 1.00 → 1.60 (E0001–E0005; all four weights changed between rounds).
* Adding general Wikipedia text at low weight (eval pages ×300) keeps nearly the eval-only
  fertility (E0017: en 1.434, hi 1.275, te 1.964, mr 1.554) and acts as a tie-breaker that
  makes the response smooth. The balancer then converged to spread 0.0044 (E0018–E0028).
* Akshara-first BPE (E0015, eval-only): 0 bad boundaries on all pages for +0.01–0.05 fertility.
* English ≤ 1.2 is expensive: English weight ×2.5 gives en 1.234 but te 2.49 (E0013);
  ×4 gives en 1.151, te 2.88 (E0014).

## B. Fourth-language comparison

Same recipe for every candidate (akshara-first, eval pages ×300 + 600K general words per
language, characters rarer than 30 dropped unless they occur on an eval page), weights
balanced automatically, then held-out fertility measured on unseen Wikipedia articles.
Results: see `experiments/SUMMARY.md` (generated).

### Selection rule (written down before stage B/C results were known)

Among recipes that give exactly 10,000 tokens, a lossless round trip, no `<unk>` and zero
in-akshara token boundaries on the evaluation pages, pick the one with the **lowest balanced
fertility level** (minimax: the worst language as good as possible). Break ties within 0.005
by held-out fertility. Only after that, fine-tune the language weights to shrink the spread.
For the fourth language, also require a substantial article (at least ~2,000 words, comparable
to the 2,453-word Telugu page), so the choice is not driven by a tiny page.

## C. Recipe ablations on en/hi/te/ur

* Fourth language chosen by the rule above: **Urdu** (balanced 1.499, lowest of the seven
  substantial pages; even the small-page candidates Gujarati 1.551, Kannada 1.593 and Sanskrit
  1.590 balance worse). Urdu is the language closest to Hindi in speech, written in a different
  script; its India article (3,486 words) is longer than the Telugu one (2,453 words).
* Midway through stage C the balancer was made more robust: it now always steps from the best
  weights found so far and halves the step after a step that made the spread worse. Runs started
  after that change (ext150, bytefb, nfkc, alpha100, evalonly) use the damped version; the
  recorded result of every run is simply the best tokenizer it trained, so levels stay comparable.
* C-bytefb (byte fallback, +256 byte tokens): balanced ≈1.518, about 0.02 worse than without.
  Its round-trip check failed because my decoder chain for that option (ByteFallback → Fuse →
  Metaspace) dropped spaces; encoding (and so fertility) was unaffected. Fixed afterwards to the
  usual Replace → ByteFallback → Fuse → Strip chain. Not used in the final tokenizer.
* C-nfkc: same level as NFC (≈1.498), so the lossless NFC is kept.
* C-plain (no akshara rule): ≈1.45 but 1,510 token boundaries inside aksharas on the pages.
* C-ew100 (pages ×100): 1.591, better held-out but worse on the pages. C-ew1000: could not be
  balanced (Telugu jumps again), spread stuck at 0.11.
* C-akfew (glue only eval-page aksharas + aksharas with weighted count ≥ 3000): 1.481, spread
  0.002, still 0 cuts on the pages. Best so far.
* C-ext150 (150K instead of 600K general words): lower mean (1.466) but could not be balanced
  (spread 0.038, worst language 1.489) and worse held-out Telugu (2.92).
* Comparison metric from here on: the worst language's fertility at the best balance reached
  (minimax), not the mean, because an unbalanceable recipe is no use for the score.

## C2. Refinement around C-akfew

* Akshara threshold 10,000 and "eval-page aksharas only" give byte-identical tokenizers (1,405
  akshara tokens; C-akfew had 1,406), so C-akfew is in effect "glue the aksharas that occur on the
  evaluation pages". The duplicate eval-only run was stopped after its first (identical) run.
* Reproducibility bug found: the ledger stored weights rounded to 6 decimals, and retraining
  from rounded weights changes a few integer word counts and hence a few merges (C-akfew-010:
  spread 0.001992 recorded vs 0.001698 retrained from the rounded weights). From now on weights
  are stored at full precision, and the released tokenizer is always produced by
  `src/make_final.py` from exact weights in `tokenizer/final_config.json` and verified afterwards.
  Earlier ledger rows remain valid as records of what was measured; only their exact retraining
  needs the unrounded weights.
* C2 results (worst language at best balance reached): ak10k ≈ akfew 1.4828 / 1.4823,
  ak1000 1.4868, pages ×200 1.503, pages ×500 1.486 (spread only 0.025 after 10 runs),
  300K general words: Telugu stuck around 1.51 (spread 0.07 after 5 runs).
* **Selected recipe (by the rule): C-akfew** — akshara-first with the aksharas of the four pages
  (plus aksharas with weighted count ≥ 3000, which adds one), pages ×300, 600K general words per
  language, characters rarer than 30 dropped unless on a page, NFC, SentencePiece-style spaces.

## D. Final stage on the selected recipe

* Plain BPE re-run with the damped balancer (C-plain2) to get a fair akshara-cost number.
* Four independent fine-tuning chains (D-ft1..4, 30 trials each, different random seeds):
  small random/directional perturbations of the four weights, keep if the spread shrinks.
  Budget fixed in advance: 120 retrains; the best tokenizer found is released.
* English trade-off: English weight ×1 … ×6 with the others fixed (D-entrade), and a balancing
  run with English pinned at 1.2 and the other three balanced (D-en12).
* C-plain2 (plain BPE, damped balancer): balanced at 1.4566 (spread 0.0016) with 1,534 cuts
  inside aksharas on the pages → the akshara rule costs ≈ 0.026 fertility.
* C2-akfew-ext300 finished with worst language 1.4768 but spread never below 0.020 in 11 runs.
  "Balanced" is made explicit here as spread ≤ 0.005 (well below the level differences being
  compared); ext300 and ew500 do not reach it, so C-akfew (1.4823 at spread 0.0020) stays selected.
* Fine-tuning chains D-ft1..4 started with σ = 0.01 and an assumed elasticity of −0.2; near
  balance a 1% weight change moves fertility by about 1% (elasticity ≈ −1), so every trial
  overshot (spreads 0.012–0.022 vs 0.00165 at the start). Stopped after ~9 trials each and
  restarted as D-ft5..8 with σ = 0.0015 and elasticity −1.0 (30 trials each).

## E. Release and independent audit

* Released tokenizer = D-ft7-030 (spread 97/203599 = 0.000476, score 2,098,958.76), retrained
  byte-identically from `tokenizer/final_config.json` by `src/make_final.py` (also with 8 instead
  of 2 threads). Budget note: the restarted chains D-ft5..8 used 4 × 31 = 124 retrains (30 trials
  + start each); together with the 44 retrains of the stopped chains D-ft1..4 that is 168.
* Note on numbers: stage B/C tables in this log quote the *mean* of the four fertilities at the
  best balance (e.g. Urdu 1.499); the website quotes the *worst* language at that point (Urdu
  1.501). Both come from the same ledger rows.
* Two plain-BPE references exist: D-base-001 (pages only, equal weights: 1,694 cuts inside
  aksharas on the pages) and C-plain2 (full recipe without the akshara rule: 1,534 cuts).
* An independent audit (separate agent, own scripts, no shared code) confirmed: exactly 10,000
  tokens; counts en 17292/11672, hi 12131/8187, te 3634/2453, ur 5166/3486; spread 97/203599;
  revision ids/titles/timestamps match the live API; texts regenerate from the raw API output;
  0 grapheme-cluster cuts on the pages; retraining reproduces the release. It also found problems,
  all fixed afterwards:
  - the evaluation texts were not downloadable from the website → now published in `site/data/eval/`
    with a copy-paste check next to the score;
  - with the `wikipedia` Python package's default "== Heading ==" format the score is ≈93,821
    (added to the sensitivity table and caveats);
  - "best attempt 1.208" understated what is reachable: English 1.189 at ×6 English weight (D-entrade2-007)
    (score 863) → caveat rewritten;
  - the English-weight sweep had started from pre-fine-tuning weights → re-run from the released
    weights as D-entrade2 (its ×1 point is the released tokenizer);
  - the manifest hash excluded the final newline → `file_sha256` (what `sha256sum` prints) added;
  - `<unk>` for characters never seen in training was not disclosed (held-out: 72/18/168/99 for
    en/hi/te/ur) → disclosed; a malformed nukta test string fixed; score decimal now exact;
  - compatibility: loads with identical counts in tokenizers 0.20.3, 0.21.0 and 0.23.1.
