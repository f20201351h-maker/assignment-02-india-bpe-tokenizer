import { BPETokenizer, countWords, SPACE } from "./bpe.js?v=d2721aaf95";

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 4) => Number(x).toFixed(d);
const int = (x) => Number(x).toLocaleString("en-US");

let D, TOK, TJ;

// ---------------------------------------------------------------- theme
(function theme() {
  const btn = $("#themeBtn");
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* storage unavailable */ }
  if (saved) document.documentElement.dataset.theme = saved;
  btn.addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
  });
})();

// ---------------------------------------------------------------- tooltip
const tip = $("#tooltip");
function showTip(html, ev) {
  tip.innerHTML = html; tip.hidden = false;
  const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.max(8, ev.clientY - tip.offsetHeight - 10);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.hidden = true; }

function bind(name, text) { document.querySelectorAll(`[data-bind="${name}"]`).forEach((el) => (el.textContent = text)); }
function bindHTML(name, html) { document.querySelectorAll(`[data-bind="${name}"], [data-bind-html="${name}"]`).forEach((el) => (el.innerHTML = html)); }

function langName(code) { return D.lang_meta[code].name; }
function langAttrs(code) { const m = D.lang_meta[code]; return `lang="${m.bcp47 || code}" dir="${m.dir || "ltr"}"`; }

function tokenChips(tokens, code) {
  return `<div class="token-out" ${code ? langAttrs(code) : ""}>` + tokens.map((t, i) => {
    const vis = esc(t).replaceAll(SPACE, `<span class="sp">${SPACE}</span>`);
    const id = TOK ? TOK.vocab.get(t) : undefined;
    return `<span class="tok${t === "<unk>" ? " tok-unk" : ""}" dir="auto" data-tip="${esc(t)}" data-id="${id ?? ""}">${vis}</span>`;
  }).join("") + "</div>";
}
document.addEventListener("mouseover", (ev) => {
  const el = ev.target.closest(".tok");
  if (!el) return;
  const id = el.dataset.id;
  showTip(`<b>${esc(el.dataset.tip)}</b><br>token id ${esc(id)}`, ev);
});
document.addEventListener("mouseout", (ev) => { if (ev.target.closest(".tok")) hideTip(); });

// ---------------------------------------------------------------- result
function renderResult() {
  const F = D.final;
  bind("langList", D.title_langs);
  bindHTML("lede", D.lede_html);
  $("#fertTiles").innerHTML = D.langs.map((c) => {
    const p = F.per_lang[c];
    const badge = c === F.min_lang ? "best (X<sub>min</sub>)" : c === F.max_lang ? "worst (X<sub>max</sub>)" : "";
    return `<div class="tile">
      ${badge ? `<span class="tile-badge">${badge}</span>` : `<span class="tile-badge-spacer" aria-hidden="true"></span>`}
      <div><span class="tile-lang">${esc(langName(c))}</span>${D.lang_meta[c].native !== langName(c) ? `<span class="tile-native" ${langAttrs(c)}>${esc(D.lang_meta[c].native)}</span>` : ""}</div>
      <div class="tile-value">${fmt(p.fertility, 6)}</div>
      <div class="tile-sub">${int(p.tokens)} tokens / ${int(p.words)} words</div>
    </div>`;
  }).join("");
  const mx = F.per_lang[F.max_lang], mn = F.per_lang[F.min_lang];
  bind("scoreShort", Number(F.score).toLocaleString("en-US", { maximumFractionDigits: 1 }));
  bind("scoreFull", `exact: ${F.score_exact}  ≈ ${F.score_decimal}`);
  bind("maxLine", `${langName(F.max_lang)}: ${mx.tokens} / ${mx.words} = ${mx.fertility.toPrecision(12)}`);
  bind("minLine", `${langName(F.min_lang)}: ${mn.tokens} / ${mn.words} = ${mn.fertility.toPrecision(12)}`);
  bind("spreadLine", `${mx.fertility.toPrecision(12)} − ${mn.fertility.toPrecision(12)} = ${F.spread.toPrecision(10)}  (= ${F.spread_exact})`);
  bind("scoreLine", `1000 / ${F.spread.toPrecision(10)} = ${F.score_decimal}`);
  bind("vocabSize", int(F.vocab_size));
  const c = F.composition;
  bind("vocabComp", `${c.special} special + ${int(c.characters)} single characters + ${int(c.akshara)} akshara-building merges + ${int(c.learned)} learned merges`);
  bind("verifiedShort", F.verification.agree ? "Reproduced independently" : "MISMATCH");
  bind("verifiedLong", F.verification.summary);
  bind("dlMeta", `sha256 ${F.tokenizer_sha256.slice(0, 16)}… · ${(F.tokenizer_bytes / 1024).toFixed(0)} KB`);
  bindHTML("caveat", D.caveat_html || "");
  $("#evalLinks").innerHTML = D.eval_downloads.map((e) =>
    `<a class="btn btn-small" href="${esc(e.href)}" download="${esc(e.lang)}.txt">${esc(langName(e.lang))} · ${esc(e.lang)}.txt</a>`).join("")
    + `<a class="btn btn-small" href="data/eval/manifest.json" download="manifest.json">manifest.json (revisions, hashes)</a>`;
  $("#checkSnippet").textContent = D.check_snippet;
}

// ---------------------------------------------------------------- languages
function renderLanguages() {
  const F = D.final;
  $("#langTable tbody").innerHTML = D.langs.map((c) => {
    const p = F.per_lang[c];
    return `<tr>
      <td><b>${esc(langName(c))}</b>${D.lang_meta[c].native !== langName(c) ? `<div class="sub" ${langAttrs(c)}>${esc(D.lang_meta[c].native)}</div>` : ""}</td>
      <td><a href="${esc(p.permanent_url)}" target="_blank" rel="noopener" ${langAttrs(c)}>${esc(p.title)}</a>
          <div class="sub">${esc(p.project)} · revision ${esc(p.revid)} · ${esc(p.rev_timestamp.slice(0, 10))}</div></td>
      <td class="num">${int(p.words)}</td><td class="num">${int(p.tokens)}</td>
      <td class="num"><b>${fmt(p.fertility, 6)}</b><div class="sub">${esc(p.fertility_exact)}</div></td>
      <td class="num">${fmt(p.heldout_fertility, 3)}<div class="sub">${int(p.heldout_words)} words</div></td>
      <td class="num">${int(p.akshara_violations)}</td>
    </tr>`;
  }).join("");
  $("#examples").innerHTML = D.langs.map((c) => {
    const ex = F.per_lang[c].example;
    return `<div class="example"><div class="example-head">${esc(langName(c))} · ${ex.tokens.length} tokens for ${countWords(ex.text)} words</div>
      ${tokenChips(ex.tokens, c)}</div>`;
  }).join("");
}

// ---------------------------------------------------------------- try it
function renderTry() {
  const input = $("#tryInput");
  $("#tryExamples").innerHTML = D.langs.map((c) => `<button type="button" class="btn btn-small" data-lang="${c}">${esc(langName(c))} example</button>`).join("")
    + `<button type="button" class="btn btn-small" data-zw="1">ZWJ / ZWNJ example</button>`;
  $("#tryExamples").addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    if (b.dataset.zw) input.value = "क्ष  क्‍ष  क्‌ष  जवाहरलाल  జవహర్‌లాల్";
    else input.value = D.final.per_lang[b.dataset.lang].example.text;
    update();
  });
  const update = () => {
    const text = input.value;
    if (!TOK) return;
    const enc = TOK.encode(text);
    const w = countWords(text);
    const unk = enc.tokens.filter((t) => t === "<unk>").length;
    $("#tryStats").textContent = `${w} words · ${enc.tokens.length} tokens · fertility ${w ? (enc.tokens.length / w).toFixed(3) : "—"}${unk ? ` · ${unk} unknown character(s)` : ""}`;
    $("#tryOut").outerHTML = `<div id="tryOut">${tokenChips(enc.tokens)}</div>`;
    const out = $("#tryOut .token-out");
    if (out) out.setAttribute("dir", "auto");
  };
  input.addEventListener("input", update);
  input.value = D.final.per_lang[D.langs[1]].example.text;
  update();
}

// ---------------------------------------------------------------- vocabulary browser
const SCRIPTS = [
  ["Latin", /[A-Za-zÀ-ɏ]/u], ["Devanagari", /[ऀ-ॿ]/u], ["Telugu", /[ఀ-౿]/u],
  ["Arabic", /[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]/u], ["Bengali", /[ঀ-৿]/u],
  ["Gurmukhi", /[਀-੿]/u], ["Tamil", /[஀-௿]/u], ["Kannada", /[ಀ-೿]/u],
  ["Malayalam", /[ഀ-ൿ]/u], ["Odia", /[଀-୿]/u], ["Gujarati", /[઀-૿]/u],
  ["Digits", /[0-9]/u],
];
function scriptOf(t) {
  for (const [name, re] of SCRIPTS) if (re.test(t)) return name;
  return "Other";
}
let VROWS = [], vPage = 0;
const PAGE = 50;
function buildVocab() {
  const built = new Map();
  TJ.model.merges.forEach((m) => { const [a, b] = Array.isArray(m) ? m : m.split(" "); built.set(a + b, [a, b]); });
  const nChars = D.final.composition.characters, nSpec = D.final.composition.special, nAk = D.final.composition.akshara;
  VROWS = TOK.idToToken.map((t, id) => {
    let kind = "learned";
    if (id < nSpec) kind = "special";
    else if (id < nSpec + nChars) kind = "char";
    else if (id < nSpec + nChars + nAk) kind = "akshara";
    return { id, t, kind, script: kind === "special" ? "Special" : scriptOf(t), len: [...t].length, from: built.get(t) };
  });
  const scripts = ["All scripts", ...new Set(VROWS.map((r) => r.script))];
  const counts = {};
  VROWS.forEach((r) => (counts[r.script] = (counts[r.script] || 0) + 1));
  $("#vocabScript").innerHTML = scripts.map((s) => `<option value="${s === "All scripts" ? "all" : s}">${s}${counts[s] ? ` (${counts[s]})` : ""}</option>`).join("");
  ["#vocabSearch", "#vocabScript", "#vocabKind"].forEach((s) => $(s).addEventListener("input", () => { vPage = 0; renderVocab(); }));
  $("#vocabPrev").addEventListener("click", () => { vPage--; renderVocab(); });
  $("#vocabNext").addEventListener("click", () => { vPage++; renderVocab(); });
  renderVocab();
}
function renderVocab() {
  const q = $("#vocabSearch").value.normalize("NFC").replaceAll(" ", SPACE);
  const sc = $("#vocabScript").value, kd = $("#vocabKind").value;
  const rows = VROWS.filter((r) =>
    (!q || r.t.includes(q)) && (sc === "all" || r.script === sc) &&
    (kd === "all" || (kd === "word" ? r.t.startsWith(SPACE) && r.t.length > 1 : r.kind === kd)));
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  vPage = Math.min(Math.max(0, vPage), pages - 1);
  const slice = rows.slice(vPage * PAGE, vPage * PAGE + PAGE);
  const kindName = { special: "special", char: "character", akshara: "akshara merge", learned: "learned merge" };
  $("#vocabTable tbody").innerHTML = slice.map((r) => `<tr>
    <td class="num">${r.id}</td>
    <td class="tokcell" dir="auto">${tokenChips([r.t]).replace('class="token-out"', 'class="token-out" style="display:inline-flex"')}</td>
    <td class="built" dir="auto">${r.from ? `${esc(r.from[0])} + ${esc(r.from[1])}` : "—"}</td>
    <td>${kindName[r.kind]}</td><td>${r.script}</td><td class="num">${r.len}</td></tr>`).join("");
  $("#vocabCount").textContent = `${int(rows.length)} of ${int(VROWS.length)} tokens`;
  $("#vocabPage").textContent = `page ${vPage + 1} of ${pages}`;
  $("#vocabPrev").disabled = vPage === 0;
  $("#vocabNext").disabled = vPage >= pages - 1;
}

// ---------------------------------------------------------------- charts (plain SVG)
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function svgEl(w, h) { return `<svg viewBox="0 0 ${w} ${h}" role="img" preserveAspectRatio="xMidYMid meet">`; }
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
  return out;
}
function attachHover(container) {
  container.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("mousemove", (ev) => showTip(el.dataset.tip, ev));
    el.addEventListener("mouseleave", hideTip);
  });
}

function chartBaseline() {
  const B = D.experiments.baseline, F = D.final;
  const rows = D.langs.map((c) => ({ c, b: B.per_lang[c], f: F.per_lang[c].fertility }));
  const all = rows.flatMap((r) => [r.b, r.f]);
  const lo = Math.floor(Math.min(...all) * 10) / 10 - 0.05, hi = Math.ceil(Math.max(...all) * 10) / 10 + 0.05;
  const W = 760, rowH = 40, top = 26, left = 110, right = 30, H = top + rowH * rows.length + 50;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  let s = svgEl(W, H);
  niceTicks(lo, hi, 6).forEach((t) => {
    s += `<line class="grid-line" x1="${x(t)}" x2="${x(t)}" y1="${top - 8}" y2="${H - 44}"/><text x="${x(t)}" y="${H - 26}" text-anchor="middle">${t.toFixed(1)}</text>`;
  });
  s += `<line x1="${x(F.max_fertility)}" x2="${x(F.max_fertility)}" y1="${top - 14}" y2="${H - 44}" stroke="${css("--series-1")}" stroke-width="1" opacity="0.5"/>`;
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    s += `<text x="${left - 12}" y="${y + 4}" text-anchor="end" class="label-strong">${esc(langName(r.c))}</text>`;
    s += `<line x1="${x(r.b)}" x2="${x(r.f)}" y1="${y}" y2="${y}" stroke="${css("--axis")}" stroke-width="2"/>`;
    s += `<circle cx="${x(r.b)}" cy="${y}" r="6" fill="${css("--surface")}" stroke="${css("--series-2")}" stroke-width="2" data-tip="${esc(langName(r.c))} baseline: ${fmt(r.b)}"/>`;
    s += `<circle cx="${x(r.f)}" cy="${y}" r="6" fill="${css("--series-1")}" stroke="${css("--surface")}" stroke-width="2" data-tip="${esc(langName(r.c))} final: ${fmt(r.f, 6)}"/>`;
  });
  s += `<text x="${x(F.max_fertility)}" y="${top - 16}" text-anchor="middle">all four ≈ ${fmt(F.max_fertility, 3)}</text>`;
  s += `<text x="${(left + W - right) / 2}" y="${H - 4}" text-anchor="middle">fertility (tokens per word)</text></svg>`;
  const el = $("#chartBaseline");
  el.innerHTML = `<div class="legend"><span class="legend-item"><span class="legend-swatch" style="border:2px solid ${css("--series-2")};width:8px;height:8px"></span>baseline: ${esc(B.label)} (score ${Number(B.score).toLocaleString("en-US", { maximumFractionDigits: 0 })})</span><span class="legend-item"><span class="legend-swatch" style="background:${css("--series-1")}"></span>final tokenizer (score ${Number(F.score).toLocaleString("en-US", { maximumFractionDigits: 0 })})</span></div>` + s;
  attachHover(el);
}

function chartFourth() {
  const rows = D.experiments.fourth.rows.slice().sort((a, b) => a.level - b.level);
  const W = 760, rowH = 28, top = 10, left = 190, right = 70, H = top + rowH * rows.length + 30;
  const lo = Math.floor(Math.min(...rows.map((r) => r.level)) * 10) / 10, hi = Math.ceil(Math.max(...rows.map((r) => r.level)) * 10) / 10 + 0.02;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  let s = svgEl(W, H);
  niceTicks(lo, hi, 5).forEach((t) => {
    s += `<line class="grid-line" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - 26}"/><text x="${x(t)}" y="${H - 8}" text-anchor="middle">${t.toFixed(1)}</text>`;
  });
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    const fill = r.chosen ? css("--series-1") : r.eligible ? css("--series-gray") : css("--surface");
    const stroke = r.chosen ? css("--surface") : r.eligible ? css("--surface") : css("--series-gray");
    const label = r.lang === "-" ? "en + hi + te only" : `${r.name}${r.eligible ? "" : " (small page)"}`;
    s += `<text x="${left - 10}" y="${y + 4}" text-anchor="end" class="${r.chosen ? "label-strong" : ""}">${esc(label)}</text>`;
    s += `<line class="grid-line" x1="${left}" x2="${x(r.level)}" y1="${y}" y2="${y}"/>`;
    s += `<circle cx="${x(r.level)}" cy="${y}" r="${r.chosen ? 7 : 6}" fill="${fill}" stroke="${stroke}" stroke-width="2"
      data-tip="${esc(label)}: balanced fertility ${fmt(r.level)}<br>page: ${r.words ? int(r.words) + " words" : "—"}${r.heldout ? `<br>held-out (worst of 4): ${fmt(r.heldout, 3)}` : ""}"/>`;
    s += `<text x="${x(r.level) + 12}" y="${y + 4}">${fmt(r.level, 3)}</text>`;
  });
  s += `<text x="${(left + W - right) / 2}" y="${H + 6}" text-anchor="middle">balanced fertility (all four languages equal), lower is better</text></svg>`;
  const el = $("#chartFourth");
  el.innerHTML = s;
  attachHover(el);
  const T = $("#fourthTable");
  T.querySelector("thead").innerHTML = `<tr><th>Fourth language</th><th class="num">Page words</th><th class="num">Balanced fertility</th><th class="num">Spread reached</th><th class="num">Held-out (worst)</th><th>Note</th></tr>`;
  T.querySelector("tbody").innerHTML = rows.map((r) => `<tr class="${r.chosen ? "hl" : ""}"><td>${esc(r.lang === "-" ? "none (en+hi+te only)" : r.name)}</td><td class="num">${r.words ? int(r.words) : "—"}</td><td class="num">${fmt(r.level)}</td><td class="num">${fmt(r.spread, 5)}</td><td class="num">${r.heldout ? fmt(r.heldout, 3) : "—"}</td><td>${esc(r.note || "")}</td></tr>`).join("");
}

function lineChart(el, series, opts) {
  const W = 760, H = 300, top = 18, left = 56, right = 110, bottom = 42;
  const xs = series.flatMap((s) => s.points.map((p) => p[0])), ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const xlo = opts.xlo ?? Math.min(...xs), xhi = opts.xhi ?? Math.max(...xs);
  const tf = opts.log ? Math.log10 : (v) => v;
  const ylo = opts.ylo ?? Math.min(...ys.map(tf)), yhi = opts.yhi ?? Math.max(...ys.map(tf));
  const x = (v) => left + ((v - xlo) / (xhi - xlo || 1)) * (W - left - right);
  const y = (v) => top + (1 - (tf(v) - ylo) / (yhi - ylo || 1)) * (H - top - bottom);
  let s = svgEl(W, H);
  const yt = opts.log ? Array.from({ length: Math.floor(yhi) - Math.ceil(ylo) + 1 }, (_, i) => 10 ** (Math.ceil(ylo) + i)) : niceTicks(ylo, yhi, 5);
  yt.forEach((t) => { s += `<line class="grid-line" x1="${left}" x2="${W - right}" y1="${y(t)}" y2="${y(t)}"/><text x="${left - 8}" y="${y(t) + 4}" text-anchor="end">${opts.yfmt ? opts.yfmt(t) : t}</text>`; });
  (opts.xticks || niceTicks(xlo, xhi, 6)).forEach((t) => { s += `<text x="${x(t)}" y="${H - bottom + 18}" text-anchor="middle">${opts.xfmt ? opts.xfmt(t) : t}</text>`; });
  s += `<line class="axis-line" x1="${left}" x2="${W - right}" y1="${H - bottom}" y2="${H - bottom}"/>`;
  (opts.hlines || []).forEach((h) => { s += `<line x1="${left}" x2="${W - right}" y1="${y(h.v)}" y2="${y(h.v)}" stroke="${css("--axis")}" stroke-width="1"/><text x="${left + 6}" y="${y(h.v) + 16}">${esc(h.label)}</text>`; });
  series.forEach((sr) => {
    if (!sr.dotsOnly) {
      const d = sr.points.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
      s += `<path d="${d}" fill="none" stroke="${sr.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    }
    sr.points.forEach((p, i) => {
      if (sr.markers === false) return;
      s += `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="${sr.dotsOnly ? 3.5 : 5}" fill="${sr.color}" stroke="${css("--surface")}" stroke-width="${sr.dotsOnly ? 1 : 2}" data-tip="${esc(sr.tips ? sr.tips[i] : `${sr.name}: ${p[1]}`)}"/>`;
    });
    if (!sr.dotsOnly) {
      const last = sr.points[sr.points.length - 1];
      s += `<text x="${x(last[0]) + 9}" y="${y(last[1]) + 4}" class="label-strong">${esc(sr.name)}</text>`;
    }
  });
  s += `<text x="${(left + W - right) / 2}" y="${H - 6}" text-anchor="middle">${esc(opts.xlabel)}</text>`;
  s += `<text transform="translate(14 ${(top + H - bottom) / 2}) rotate(-90)" text-anchor="middle">${esc(opts.ylabel)}</text></svg>`;
  el.innerHTML = `<div class="legend">${series.map((sr) => `<span class="legend-item"><span class="legend-line" style="background:${sr.color}"></span>${esc(sr.name)}</span>`).join("")}</div>` + s;
  attachHover(el);
}

function chartTradeoff() {
  const P = D.experiments.tradeoff.points;
  lineChart($("#chartTradeoff"), [
    { name: "English", color: css("--series-1"), points: P.map((p) => [p.en_weight, p.en]), tips: P.map((p) => `English weight ×${p.en_weight}: English ${fmt(p.en)}`) },
    { name: "worst other language", color: css("--series-2"), points: P.map((p) => [p.en_weight, p.worst_other]), tips: P.map((p) => `English weight ×${p.en_weight}: worst other (${esc(langName(p.worst_lang))}) ${fmt(p.worst_other)}`) },
  ], { xlabel: "English sampling weight relative to the balanced setting", ylabel: "fertility", hlines: [{ v: 1.2, label: "1.2 target for English" }], ylo: 1.0, xfmt: (t) => `×${t}` });
}

function chartTrace() {
  const P = D.experiments.trace.points;
  let best = Infinity;
  const bestPts = P.map((p) => { best = Math.min(best, p.spread); return [p.step, best]; });
  const ys = P.map((p) => Math.log10(p.spread));
  lineChart($("#chartTrace"), [
    { name: "each run", color: css("--series-gray"), dotsOnly: true, points: P.map((p) => [p.step, p.spread]),
      tips: P.map((p) => `run ${p.step} (${esc(p.id)}): spread ${p.spread.toPrecision(4)}, score ${Math.round(1000 / p.spread).toLocaleString("en-US")}`) },
    { name: "best so far", color: css("--series-1"), markers: false, points: bestPts },
  ], { xlabel: "run (balancing, then fine-tuning trials)", ylabel: "spread (log scale)", log: true,
       ylo: Math.floor(Math.min(...ys)), yhi: Math.ceil(Math.max(...ys)),
       yfmt: (t) => (t >= 0.001 ? String(+t.toPrecision(3)) : t.toExponential(0)) });
}

function simpleTable(sel, head, rows) {
  const T = $(sel);
  T.querySelector("thead").innerHTML = `<tr>${head.map((h) => `<th class="${h.num ? "num" : ""}">${h.label}</th>`).join("")}</tr>`;
  T.querySelector("tbody").innerHTML = rows.map((r) => `<tr class="${r._hl ? "hl" : ""}">${head.map((h) => `<td class="${h.num ? "num" : ""}">${r[h.key] ?? "—"}</td>`).join("")}</tr>`).join("");
}

function renderExperiments() {
  const E = D.experiments;
  bindHTML("expIntro", E.intro_html);
  bindHTML("baselineNote", E.baseline.note_html);
  bindHTML("fourthNote", E.fourth.note_html);
  bindHTML("tradeoffNote", E.tradeoff.note_html);
  bindHTML("ablationNote", E.ablations.note_html);
  bindHTML("traceNote", E.trace.note_html);
  bindHTML("sensNote", E.sensitivity.note_html);
  chartBaseline(); chartFourth(); chartTradeoff(); chartTrace();
  simpleTable("#ablationTable", [
    { key: "name", label: "Recipe" }, { key: "level", label: "Balanced fertility", num: true },
    { key: "heldout", label: "Held-out (worst)", num: true }, { key: "akv", label: "Cuts inside aksharas (eval / held-out)", num: true },
    { key: "note", label: "Observation" }], E.ablations.rows.map((r) => ({ ...r, level: fmt(r.level), heldout: r.heldout ? fmt(r.heldout, 3) : "—", akv: `${int(r.akv_eval)} / ${int(r.akv_heldout)}`, note: esc(r.note), name: esc(r.name), _hl: r.chosen })));
  simpleTable("#sensTable", [{ key: "variant", label: "Text used" },
    ...D.langs.map((c) => ({ key: c, label: esc(langName(c)), num: true })),
    { key: "spread", label: "Spread", num: true }, { key: "score", label: "Score", num: true }],
    E.sensitivity.rows.map((r) => ({ variant: `${esc(r.variant)}<div class="sub">${esc(r.desc)}</div>`, ...Object.fromEntries(D.langs.map((c) => [c, fmt(r.per_lang[c], 4)])), spread: r.spread.toPrecision(4), score: Math.round(r.score).toLocaleString("en-US"), _hl: r.primary })));
}

function renderMethod() { bindHTML("method", D.method_html); }

function renderRepro() {
  const R = D.repro;
  simpleTable("#reproTable", [{ key: "lang", label: "Language" }, { key: "page", label: "Page (revision)" }, { key: "retrieved", label: "Retrieved (UTC)" }, { key: "sha", label: "Text file SHA-256 (sha256sum)" }, { key: "words", label: "Words", num: true }],
    D.langs.map((c) => { const p = D.final.per_lang[c]; return { lang: esc(langName(c)), page: `<a href="${esc(p.permanent_url)}" target="_blank" rel="noopener">${esc(p.project)} · ${esc(p.revid)}</a><div class="sub"><a href="${esc(p.eval_file)}" download>${esc(c)}.txt</a></div>`, retrieved: esc(p.retrieved_utc), sha: `<code>${esc(p.file_sha256)}</code>`, words: int(p.words) }; }));
  $("#reproInfo").innerHTML = R.blocks.map((b) => `<div><h3>${esc(b.title)}</h3>${b.html || ""}${b.code ? `<pre>${esc(b.code)}</pre>` : ""}</div>`).join("");
}

// ---------------------------------------------------------------- boot
async function boot() {
  try {
    const [d, tj] = await Promise.all([
      fetch("data/site_data.json", { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error("site_data.json " + r.status); return r.json(); }),
      fetch("tokenizer.json", { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error("tokenizer.json " + r.status); return r.json(); }),
    ]);
    D = d; TJ = tj; TOK = new BPETokenizer(tj);
  } catch (e) {
    const el = $("#loadError"); el.hidden = false;
    el.textContent = "Could not load the data files (" + e.message + "). If you opened index.html directly from disk, serve the folder over HTTP instead (e.g. python -m http.server).";
    return;
  }
  renderResult(); renderLanguages(); renderTry(); buildVocab(); renderMethod(); renderExperiments(); renderRepro();
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(renderExperiments, 200); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderExperiments);
  new MutationObserver(renderExperiments).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
}
boot();
