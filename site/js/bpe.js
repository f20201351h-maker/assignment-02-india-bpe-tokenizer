// Independent re-implementation of this project's tokenizer.json pipeline.
// It does NOT use the Hugging Face library; it reads the vocabulary and merge list and
// replays them exactly as the BPE algorithm describes:
//   1. normalise: Unicode NFC, every whitespace run -> one space, trim
//   2. pre-tokenise: spaces become "▁", a "▁" is put in front, text is cut before every "▁"
//   3. BPE: start from single characters, repeatedly merge the adjacent pair with the
//      lowest merge rank (leftmost first) until no listed merge applies.
// Works in the browser and in Node (ES module).

export const SPACE = "▁";
// Unicode White_Space characters (the set used by the Rust tokenizer's \s and trim)
const WS = "\\t\\n\\v\\f\\r \\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WS_RUN = new RegExp(`[${WS}]+`, "gu");
const WS_EDGE = new RegExp(`^[${WS}]+|[${WS}]+$`, "gu");

export class BPETokenizer {
  constructor(tj) {
    const m = tj.model;
    if (m.type !== "BPE") throw new Error("not a BPE model");
    this.vocab = new Map(Object.entries(m.vocab));
    this.idToToken = [];
    for (const [t, i] of this.vocab) this.idToToken[i] = t;
    this.ranks = new Map();
    m.merges.forEach((mg, i) => {
      const [a, b] = Array.isArray(mg) ? mg : mg.split(" ");
      this.ranks.set(a + "\u0000" + b, i);
    });
    this.unk = m.unk_token;
    this.byteFallback = !!m.byte_fallback;
    this.cache = new Map();
    this._checkPipeline(tj);
  }

  _checkPipeline(tj) {
    // Refuse to run if the file uses a pipeline this re-implementation does not cover.
    const n = JSON.stringify(tj.normalizer);
    if (!n.includes('"NFC"') || !n.includes('"Strip"')) throw new Error("unexpected normalizer: " + n);
    const p = tj.pre_tokenizer;
    if (p.type !== "Metaspace" || p.replacement !== SPACE || p.prepend_scheme !== "always" || !p.split)
      throw new Error("unexpected pre_tokenizer: " + JSON.stringify(p));
    if (tj.post_processor) throw new Error("unexpected post_processor");
    if (tj.model.ignore_merges) throw new Error("ignore_merges not supported");
  }

  normalize(text) {
    return text.normalize("NFC").replace(WS_RUN, " ").replace(WS_EDGE, "");
  }

  pretokenize(norm) {
    if (!norm) return [];
    let s = norm.replaceAll(" ", SPACE);
    if (!s.startsWith(SPACE)) s = SPACE + s;
    const out = [];
    let cur = "";
    for (const ch of s) {
      if (ch === SPACE && cur) { out.push(cur); cur = ""; }
      cur += ch;
    }
    if (cur) out.push(cur);
    return out;
  }

  bpe(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    const syms = [];
    for (const ch of word) {
      if (this.vocab.has(ch)) syms.push(ch);
      else if (this.byteFallback) {
        for (const b of new TextEncoder().encode(ch)) syms.push(`<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`);
      } else syms.push(this.unk);
    }
    while (syms.length > 1) {
      let best = Infinity, bi = -1;
      for (let i = 0; i < syms.length - 1; i++) {
        const r = this.ranks.get(syms[i] + "\u0000" + syms[i + 1]);
        if (r !== undefined && r < best) { best = r; bi = i; }
      }
      if (bi < 0) break;
      syms.splice(bi, 2, syms[bi] + syms[bi + 1]);
    }
    if (this.cache.size < 200000) this.cache.set(word, syms);
    return syms;
  }

  encode(text) {
    const tokens = [], words = [];
    this.pretokenize(this.normalize(text)).forEach((w, wi) => {
      for (const t of this.bpe(w)) { tokens.push(t); words.push(wi); }
    });
    return { tokens, ids: tokens.map((t) => this.vocab.get(t)), wordIndex: words };
  }

  decode(ids) {
    const bytes = [];
    let out = "";
    const flush = () => { if (bytes.length) { out += new TextDecoder().decode(new Uint8Array(bytes)); bytes.length = 0; } };
    for (const id of ids) {
      const t = this.idToToken[id];
      const mb = /^<0x([0-9A-F]{2})>$/.exec(t);
      if (mb && this.byteFallback) { bytes.push(parseInt(mb[1], 16)); continue; }
      flush();
      out += t;
    }
    flush();
    out = out.replaceAll(SPACE, " ");
    return out.startsWith(" ") ? out.slice(1) : out;
  }
}

// Word-count rule used for the fertility denominator: whitespace-separated chunks.
export function countWords(text) {
  return text.split(new RegExp(`[${WS}]+`, "u")).filter((w) => w.length > 0).length;
}
