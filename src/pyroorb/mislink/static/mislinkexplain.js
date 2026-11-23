"use strict";
// PyRoOrb web UI -- the local explanation of one mislinkage score, shared by the fit page's
// note (mislinknote.js) and the watch list (/mislinkage-candidates).
//
// The score comes from a gradient-boosted tree model (src/pyroorb/mislink/score.py). Its explanation is
// exact TreeSHAP in log-odds: base + the sum of every feature's contribution = the score's
// log-odds, so the bars below ADD UP to the score. Red pushes toward "chimera" (tracklets of
// more than one body), green toward "one body". The largest contributions are drawn one by
// one; the rest are summed into one row so nothing is hidden from the total.
//
// window.MislinkExplain.render(el, expl, descriptions, opts)
//   expl: {base, contributions: [{feature, value, shap}] or [[feature, value, shap]]}
//   descriptions: {feature: {label, unit, text}}
//   opts.top: rows drawn individually (default 8)

(function () {
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  function fmtValue(v, unit) {
    if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
    const a = Math.abs(v);
    const s = a === 0 ? "0" : a >= 1000 ? v.toFixed(0) : a >= 10 ? v.toFixed(1)
      : a >= 0.01 ? v.toFixed(3) : v.toExponential(1);
    return unit ? `${s} ${unit}` : s;
  }

  function rows(expl) {
    return (expl.contributions || []).map((c) => Array.isArray(c)
      ? { feature: c[0], value: c[1], shap: c[2] } : c)
      .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  }

  function render(el, expl, descriptions, opts) {
    if (!el || !expl) return;
    const top = (opts && opts.top) || 8;
    const desc = descriptions || {};
    const all = rows(expl);
    const shown = all.slice(0, top);
    const rest = all.slice(top);
    const restSum = rest.reduce((s, r) => s + r.shap, 0);
    const total = (expl.base || 0) + all.reduce((s, r) => s + r.shap, 0);
    const scale = Math.max(0.5, ...shown.map((r) => Math.abs(r.shap)), Math.abs(restSum));

    const bar = (v) => {
      const w = Math.min(50, (Math.abs(v) / scale) * 50);
      const cls = v >= 0 ? "mx-pos" : "mx-neg";
      const left = v >= 0 ? 50 : 50 - w;
      return `<div class="mx-track"><div class="mx-axis"></div>` +
             `<div class="mx-bar ${cls}" style="left:${left}%;width:${w}%"></div></div>`;
    };
    const line = (label, valueText, v, tip) =>
      `<div class="mx-row"${tip ? ` title="${esc(tip)}"` : ""}>` +
      `<div class="mx-name">${label}</div><div class="mx-val">${valueText}</div>` +
      bar(v) + `<div class="mx-shap ${v >= 0 ? "mx-pos-t" : "mx-neg-t"}">` +
      `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}</div></div>`;

    let html = `<div class="mx-head"><span>Feature</span><span>Value</span>` +
      `<span class="mx-dir"><span class="mx-neg-t">← one body</span>` +
      `<span class="mx-pos-t">chimera →</span></span><span>log-odds</span></div>`;
    html += `<div class="mx-row mx-base" title="The model's average log-odds before this object's features are read">` +
      `<div class="mx-name">Starting point</div><div class="mx-val"></div>` +
      `<div class="mx-track"></div><div class="mx-shap">${(expl.base || 0).toFixed(2)}</div></div>`;
    for (const r of shown) {
      const d = desc[r.feature] || {};
      html += line(`${esc(d.label || r.feature)} <code class="mx-key">${esc(r.feature)}</code>`,
                   esc(fmtValue(r.value, d.unit)), r.shap, d.text || "");
    }
    if (rest.length) {
      html += line(`${rest.length} other features`, "", restSum,
                   rest.map((r) => `${(desc[r.feature] || {}).label || r.feature} ` +
                                   `${r.shap >= 0 ? "+" : ""}${r.shap.toFixed(2)}`).join("; "));
    }
    html += `<div class="mx-row mx-total"><div class="mx-name">Score</div><div class="mx-val"></div>` +
      `<div class="mx-track"></div><div class="mx-shap">${total.toFixed(2)} → ` +
      `<b>${sigmoid(total).toFixed(3)}</b></div></div>`;
    html += `<details class="mx-gloss"><summary>What these features measure</summary><dl>` +
      shown.map((r) => {
        const d = desc[r.feature] || {};
        return `<dt>${esc(d.label || r.feature)} <code>${esc(r.feature)}</code></dt>` +
               `<dd>${esc(d.text || "")}</dd>`;
      }).join("") + `</dl></details>`;
    el.innerHTML = `<div class="mx">${html}</div>`;
  }

  window.MislinkExplain = { render, fmtValue, esc };
})();
