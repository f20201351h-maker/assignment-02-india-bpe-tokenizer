// Independent verification in Node.js (no Hugging Face code involved).
// usage: node verify/verify_js.mjs <tokenizer.json> <eval_dir> <lang,lang,lang,lang> [out.json]
import fs from "node:fs";
import crypto from "node:crypto";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const { BPETokenizer, countWords } = await import(pathToFileURL(path.join(here, "..", "site", "js", "bpe.js")).href);

const [tokPath, evalDir, langArg, outPath] = process.argv.slice(2);
const raw = fs.readFileSync(tokPath, "utf8");
const tj = JSON.parse(raw);
const tok = new BPETokenizer(tj);
const manifest = JSON.parse(fs.readFileSync(path.join(evalDir, "manifest.json"), "utf8"));
const seg = new Intl.Segmenter("und", { granularity: "grapheme" });

// vocabulary checks
const ids = [...tok.vocab.values()].sort((a, b) => a - b);
const vocabContiguous = ids.every((v, i) => v === i);
let mergesValid = true;
for (const mg of tj.model.merges) {
  const [a, b] = Array.isArray(mg) ? mg : mg.split(" ");
  if (!tok.vocab.has(a) || !tok.vocab.has(b) || !tok.vocab.has(a + b)) { mergesValid = false; break; }
}

const res = {
  implementation: "verify/verify_js.mjs + site/js/bpe.js (independent JS BPE)",
  node: process.version, icu_unicode: process.versions.unicode,
  tokenizer_sha256: crypto.createHash("sha256").update(raw).digest("hex"),
  vocab_size: tok.vocab.size, added_tokens: (tj.added_tokens || []).length,
  vocab_ids_contiguous: vocabContiguous, n_merges: tj.model.merges.length, merges_valid: mergesValid,
  languages: {},
};

for (const lang of langArg.split(",")) {
  const text = fs.readFileSync(path.join(evalDir, `${lang}.txt`), "utf8").replace(/\n$/, "");
  const sha = crypto.createHash("sha256").update(text, "utf8").digest("hex");
  const enc = tok.encode(text);
  const words = countWords(text);
  // grapheme check: every boundary between two tokens of the same word must be a grapheme boundary
  const norm = tok.normalize(text);
  const pre = tok.pretokenize(norm);
  let gBad = 0, unk = 0;
  let k = 0;
  pre.forEach((w) => {
    const pieces = tok.bpe(w);
    k += pieces.length;
    const gb = new Set();
    let pos = 0;
    for (const s of seg.segment(w)) { pos = s.index; gb.add(pos); }
    let off = 0;
    for (let i = 0; i < pieces.length - 1; i++) {
      off += pieces[i].length;           // UTF-16 offset, same unit as Intl.Segmenter
      if (!gb.has(off)) gBad++;
    }
    unk += pieces.filter((p) => p === tok.unk).length;
  });
  const roundtrip = tok.decode(enc.ids) === norm;
  res.languages[lang] = {
    clean_text_sha256: sha, sha_matches_manifest: sha === manifest.pages[lang].clean_text_sha256,
    words, tokens: enc.ids.length, fertility: enc.ids.length / words,
    unk_tokens: unk, grapheme_boundary_violations: gBad, roundtrip_exact: roundtrip,
  };
}
const f = Object.entries(res.languages).map(([l, v]) => [l, v.fertility]);
f.sort((a, b) => a[1] - b[1]);
res.min = { lang: f[0][0], fertility: f[0][1] };
res.max = { lang: f[f.length - 1][0], fertility: f[f.length - 1][1] };
res.spread = res.max.fertility - res.min.fertility;
res.score = 1000 / res.spread;
const out = JSON.stringify(res, null, 1);
if (outPath) fs.writeFileSync(outPath, out);
console.log(out);
