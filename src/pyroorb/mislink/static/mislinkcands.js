"use strict";
// PyRoOrb -- the mislinkage watch list (/mislinkage-candidates).
//
// Reads GET /api/mislinkage_candidates: the committed list (src/pyroorb/mislink/candidates.json) plus
// the feature glossary. A sortable, filterable table; clicking a row opens its local
// explanation (mislinkexplain.js) with links to fit the object here and to the MPC's page.
// The open object is kept in the address (#2023 RE178) so a link can point straight at one.

(function () {
  const { render, fmtValue, esc } = window.MislinkExplain;
  const $ = (id) => document.getElementById(id);
  let data = null;
  let sortKey = "score", sortAsc = false;
  let openDesig = decodeURIComponent((location.hash || "").slice(1)) || null;

  const flaggedNow = (c) => !c.now || (c.now.status === "ok" && c.now.score >= data.threshold);
  const nowText = (c) => {
    if (!c.now) return "";
    if (c.now.status === "ok") return c.now.score.toFixed(3);
    return { "multi-opposition": "now multi-opp", "out-of-scope-nights": `now ${c.now.nights} nights`,
             "too-few-obs": "obs gone", "single-detection-nights": "single dets",
             "fit-failed": "fit failed", "unavailable": "", "error": "error" }[c.now.status]
      || c.now.status;
  };
  const topReason = (c) => {
    const r = c.explanation.contributions[0];
    if (!r) return "";
    const d = data.descriptions[r[0]] || {};
    return `${d.label || r[0]} ${fmtValue(r[1], d.unit)}`;
  };

  const COLS = [
    { key: "designation", label: "Object", get: (c) => c.designation },
    { key: "score", label: "Score", num: true, get: (c) => c.score },
    { key: "now", label: "Now", num: true, get: (c) => (c.now && c.now.status === "ok") ? c.now.score : -1 },
    { key: "year", label: "Year", num: true, get: (c) => c.year },
    { key: "nights", label: "Nights", num: true, get: (c) => c.nights },
    { key: "n_tracklets", label: "Trk/Stn", num: true, get: (c) => c.n_tracklets },
    { key: "arc_days", label: "Arc d", num: true, get: (c) => c.arc_days },
    { key: "gap1", label: "Gap d", num: true, get: (c) => c.gap1 },
    { key: "rms_n", label: "rms_n", num: true, get: (c) => c.rms_n },
    { key: "v_avg", label: "V", num: true, get: (c) => c.v_avg },
    { key: "max_dH", label: "max dH", num: true, get: (c) => c.max_dH },
    { key: "mpc_rules_accept", label: "ITF rules", get: (c) => c.mpc_rules_accept ? 1 : 0 },
    { key: "reason", label: "Largest reason", get: (c) => topReason(c) },
  ];

  // the server's grading (mislink.score.TIERS), highest bound first
  const tierOf = (x) => ((data.tiers || []).find((t) => x >= t.min) || {}).verdict || "";

  const f = (v, d) => (v === null || v === undefined) ? "" : Number(v).toFixed(d);

  function cell(c, col) {
    switch (col.key) {
      case "designation": {
        const b = c.label ? `<span class="badge${c.label === "reviewed chimera" || c.label === "known mislinkage" ? " chim" : ""}">${esc(c.label)}</span>` : "";
        const g = flaggedNow(c) ? "" : `<span class="badge gone">no longer flagged</span>`;
        return `<td class="mono">${esc(c.designation)}${b}${g}</td>`;
      }
      case "score": return `<td class="num score-hi" title="${c.score_is_out_of_fold ? "scored by a model trained without this object's label" : ""}">${f(c.score, 3)}` +
        `<div class="tier">${esc(tierOf(c.score))}</div></td>`;
      case "now": return `<td class="num">${esc(nowText(c))}</td>`;
      case "n_tracklets": return `<td class="num">${c.n_tracklets ?? ""}/${c.n_stations ?? ""}</td>`;
      case "arc_days": case "gap1": return `<td class="num">${f(c[col.key], 1)}</td>`;
      case "rms_n": case "max_dH": return `<td class="num">${f(c[col.key], 2)}</td>`;
      case "v_avg": return `<td class="num">${f(c.v_avg, 1)}</td>`;
      case "mpc_rules_accept": return `<td>${c.mpc_rules_accept ? "accept" : "reject"}</td>`;
      case "reason": return `<td class="reason">${esc(topReason(c))}</td>`;
      default: return `<td class="num">${esc(c[col.key] ?? "")}</td>`;
    }
  }

  function filtered() {
    const q = $("q").value.trim().toLowerCase().replace(/\s+/g, " ");
    const y = $("year").value, n = $("nights").value;
    const col = COLS.find((c) => c.key === sortKey);
    return data.candidates
      .filter((c) => (!q || c.designation.toLowerCase().includes(q))
        && (!y || String(c.year) === y) && (!n || String(c.nights) === n)
        && (!$("hideGone").checked || flaggedNow(c))
        && (!$("rulesOnly").checked || c.mpc_rules_accept))
      .sort((a, b) => {
        const va = col.get(a), vb = col.get(b);
        const r = typeof va === "string" ? va.localeCompare(vb) : (va ?? -Infinity) - (vb ?? -Infinity);
        return sortAsc ? r : -r;
      });
  }

  function detailHtml(c) {
    const enc = encodeURIComponent(c.designation);
    const facts = [
      ["score", f(c.score, 3) + (c.score_is_out_of_fold ? " (out-of-fold)" : "") + ` — ${tierOf(c.score)}`],
      ["now", nowText(c) || "not rescored"],
      ["nights", c.nights], ["observations", c.n_obs],
      ["tracklets / stations", `${c.n_tracklets}/${c.n_stations}`],
      ["arc", `${f(c.arc_days, 1)} d`], ["gaps", `${f(c.gap1, 1)}, ${f(c.gap2, 1)} d`],
      ["a / e / q", `${f(c.a, 2)} / ${f(c.e, 2)} / ${f(c.q, 2)}`],
      ["rms_n", f(c.rms_n, 2)], ["rejected", c.n_rejected],
    ].map(([k, v]) => `<span>${esc(k)} <b>${esc(v)}</b></span>`).join("");
    const note = c.score_is_out_of_fold
      ? `<div class="note">This object was judged by hand and that judgement trained the final model, so the ` +
        `score above comes from a model trained without it; the explanation below is the final model's ` +
        `(its score ${c.explanation.model_score.toFixed(3)}).</div>` : "";
    return `<div class="facts">${facts}</div>` +
      `<div class="links"><a href="/?object=${enc}" target="_blank" rel="noopener">Fit in PyRoOrb ↗</a>` +
      `<a href="https://minorplanetcenter.net/db_search/show_object?object_id=${enc}" target="_blank" rel="noopener">MPC page ↗</a></div>` +
      note + `<div class="mx-holder"></div>`;
  }

  function draw() {
    const rows = filtered();
    $("count").textContent = `${rows.length} of ${data.candidates.length} shown`;
    $("head").innerHTML = COLS.map((c) =>
      `<th data-key="${c.key}" class="${c.num ? "num" : ""}${c.key === sortKey ? " sorted" + (sortAsc ? " asc" : "") : ""}">${esc(c.label)}</th>`).join("");
    const html = [];
    for (const c of rows) {
      const isOpen = c.designation === openDesig;
      html.push(`<tr class="row${isOpen ? " open" : ""}" data-d="${esc(c.designation)}">${COLS.map((col) => cell(c, col)).join("")}</tr>`);
      if (isOpen) html.push(`<tr class="detail"><td colspan="${COLS.length}">${detailHtml(c)}</td></tr>`);
    }
    $("body").innerHTML = html.join("");
    const holder = $("body").querySelector("tr.detail .mx-holder");
    if (holder) {
      const c = data.candidates.find((x) => x.designation === openDesig);
      render(holder, c.explanation, data.descriptions, { top: 10 });
    }
  }

  function wire() {
    $("head").addEventListener("click", (e) => {
      const th = e.target.closest("th");
      if (!th) return;
      if (th.dataset.key === sortKey) sortAsc = !sortAsc;
      else { sortKey = th.dataset.key; sortAsc = sortKey === "designation"; }
      draw();
    });
    $("body").addEventListener("click", (e) => {
      if (e.target.closest("a")) return;
      const tr = e.target.closest("tr.row");
      if (!tr) return;
      openDesig = openDesig === tr.dataset.d ? null : tr.dataset.d;
      history.replaceState(null, "", location.pathname + (openDesig ? "#" + encodeURIComponent(openDesig) : ""));
      draw();
    });
    for (const id of ["q", "year", "nights", "hideGone", "rulesOnly"]) {
      $(id).addEventListener("input", draw);
    }
  }

  function cards() {
    const cs = data.candidates;
    const still = cs.filter(flaggedNow).length;
    const items = [
      ["Candidates", cs.length], ["Still flagged now", still],
      ["ITF rules accept", cs.filter((c) => c.mpc_rules_accept).length],
      ["Arcs scored", data.n_scored],
    ];
    $("cards").innerHTML = items.map(([k, v]) =>
      `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  }

  fetch("/api/mislinkage_candidates").then((r) => r.json()).then((d) => {
    if (!d || !d.ok) throw new Error((d && d.error) || "no list");
    data = d;
    $("thr").textContent = d.threshold;
    $("when").textContent = `census ${d.census_date} · exported ${d.exported_at.slice(0, 10)}`;
    const years = [...new Set(d.candidates.map((c) => c.year))].sort();
    $("year").innerHTML += years.map((y) => `<option>${y}</option>`).join("");
    if (openDesig && !d.candidates.some((c) => c.designation === openDesig)) openDesig = null;
    cards();
    wire();
    draw();
    const open = $("body").querySelector("tr.open");
    if (open) open.scrollIntoView({ block: "center" });
  }).catch((e) => {
    $("err").hidden = false;
    $("err").textContent = `Could not load the list: ${e.message}`;
    $("when").textContent = "";
  });
})();
