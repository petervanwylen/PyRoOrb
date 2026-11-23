"use strict";
// PyRoOrb web UI -- mislinkage (chimera) score of a 3-5 night single-opposition arc.
// Vanilla JS, no build step. Loaded AFTER app.js, fitjob.js and mislinkexplain.js.
//
// Which arc is scored (POST /api/mislinkage, src/pyroorb/mislink/score.py):
//   * a fit of PASTED or ADDED observations (payload meta.source "pasted"/"added"): exactly the
//     observations on this page, all of them as reported ({session|fit_key, page: true});
//   * a fit fetched by designation, single-opposition with one provisional designation: the
//     object's PUBLISHED observations (the SBN replica) -- what the watch list scores.
//     Anything else (numbered, multi-opposition, a linkage of designations) is not asked.
// The answer is always shown: a score with an Explain toggle (mislinkexplain.js); or one muted
// line saying why the arc is not scored (not 3-5 nights, too few observations, ...); or, when
// the request fails, the reason and a retry. The request goes through the fit pool, so a
// hand-off is polled out (FitJob.settle) and a failure is retried once before it is shown.
// Display only: nothing in the fit acts on it.
//
// Listens on the "pyroorb:render" event app.js's render() dispatches (like d2note.js).

(function () {
  const PROVISIONAL = /^\d{4} [A-Z]{2}\d*$/;
  const RETRY_MS = 1500;
  const cache = new Map();          // request key -> answered response (ok: true)
  let current = null;               // key of the arc the box is showing: a late answer for a
                                    // previous object never paints over the current one
  let open = false;

  function ensureBox() {
    let box = document.getElementById("mislinkNote");
    if (!box) {
      const anchor = document.getElementById("d2Note") || document.getElementById("constraintBanner");
      if (!anchor) return null;
      box = document.createElement("div");
      box.id = "mislinkNote";
      box.hidden = true;
      anchor.insertAdjacentElement("afterend", box);
    }
    return box;
  }

  const esc = (s) => window.MislinkExplain ? window.MislinkExplain.esc(s) : String(s);

  // What to ask for this payload, or null.
  function request(p) {
    if (!p) return null;
    const src = p.meta && p.meta.source;
    if (src === "pasted" || src === "added") {
      if (!p.session && !p.fit_key) return null;
      const res = p.residuals || [];
      const jds = res.map((r) => r.jd_utc).filter((x) => x != null);
      const desig = (p.designations || [])[0] || p.object || "pasted object";
      return { key: `page:${p.object}:${res.length}:${Math.min(...jds)}:${Math.max(...jds)}`,
               body: { designation: desig, session: p.session || null,
                       fit_key: p.fit_key || null, page: true } };
    }
    if (p.combined || !p.summary || p.summary.n_oppositions !== 1) return null;
    const ds = p.designations || [];
    if (ds.length !== 1 || !PROVISIONAL.test(ds[0])) return null;
    return { key: `pub:${ds[0]}`, body: { designation: ds[0] } };
  }

  const REASON = {
    "out-of-scope-nights": (r) => `${r.nights} nights; the model covers 3–5`,
    "too-few-obs": (r) => `${r.n_obs} observations; at least 6 are needed`,
    "multi-opposition": (r) => `the arc spans ${Math.round(r.arc_days)} d, more than one opposition`,
    "single-detection-nights": () => "fewer than 3 nights hold two or more detections from one station",
    "fit-failed": () => "the orbit fit of these observations failed",
    "unavailable": () => "the published observations are not reachable from this server",
  };
  const srcText = (r) => r.source === "page" ? "the observations on this page" : "the published observations";

  function muted(box, html) {
    box.className = "mislink-muted";
    box.innerHTML = html;
    box.hidden = false;
  }

  function paint(box, r, req) {
    if (!r || !r.ok) {
      muted(box, `Mislinkage score unavailable: ${esc((r && r.error) || "no answer from the server")} ` +
                 `<button type="button" class="mx-retry">retry</button>`);
      box.querySelector(".mx-retry").addEventListener("click", () => {
        muted(box, "Mislinkage score: asking again…");
        ask(req, 0).then((x) => { if (current === req.key) paint(box, x, req); });
      });
      return;
    }
    if (r.status !== "ok") {
      const why = (REASON[r.status] || (() => r.error || r.status))(r);
      muted(box, `Mislinkage score: not scored — ${esc(srcText(r))}: ${esc(why)}.`);
      return;
    }
    // the server grades the score (mislink.score.TIERS); colour follows the grade, and an
    // out-of-domain arc is never coloured as a warning
    const level = r.in_domain ? (r.level || (r.score >= r.threshold ? "likely" : "none")) : "none";
    const alarm = level === "very-likely" || level === "likely";
    const warn = alarm || level === "possible";
    box.className = "constraint-warn" + (alarm ? " severe" : warn ? "" : " mx-calm");
    const gaps = (r.gaps_days || []).filter((g) => g).map((g) => g.toFixed(1)).join(", ");
    const verdict = !r.in_domain
      ? "outside the model's training domain (q < 1.3 au or a > 5.5 au): not meaningful here"
      : r.verdict || (alarm ? "likely mislinkage" : "no sign of a mislinkage");
    const skipped = r.n_skipped ? ` ${r.n_skipped} space-based or roving observations left out.` : "";
    // detections ~2 h apart (a survey cadence) do not form 120-min tracklets, so such an arc is
    // scored on one tracklet per station-night (mislink.score, tracklet_rule)
    const rule = r.tracklet_rule === "station-night"
      ? " Detections more than 120 min apart: tracklets taken per station-night, not the 120-min rule the model was trained on." : "";
    box.innerHTML =
      `<div class="cw-head">${warn ? "⚠" : "ℹ"} Mislinkage score ` +
      `<b>${r.score.toFixed(3)}</b> — ${esc(verdict)} ` +
      `<button type="button" class="mx-toggle">${open ? "Hide" : "Explain"}</button></div>` +
      `<div class="cw-note">How strongly a tree model reads these ${r.nights} nights ` +
      `(${r.n_tracklets} tracklets, ${r.n_stations} stations; largest gaps ${esc(gaps)} d) ` +
      `as holding more than one body, from ${esc(srcText(r))}.${skipped}${rule} ` +
      `A ranking, not a calibrated probability. ` +
      `<a href="/mislinkage-candidates" target="_blank" rel="noopener">Watch list ↗</a></div>` +
      `<div class="mx-holder"${open ? "" : " hidden"}></div>`;
    const holder = box.querySelector(".mx-holder");
    if (open && window.MislinkExplain) {
      window.MislinkExplain.render(holder, r.explanation, r.descriptions);
    }
    box.querySelector(".mx-toggle").addEventListener("click", () => { open = !open; paint(box, r, req); });
    box.hidden = false;
  }

  async function once(req) {
    try {
      const x = await fetch("/api/mislinkage", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      let r = JSON.parse(await x.text());
      // past the pool's inline deadline the answer is a job to poll, not the score
      if (window.FitJob && window.FitJob.isPending(r)) r = await window.FitJob.settle(r);
      return r;
    } catch (e) {
      return { ok: false, error: e.message || "request failed" };
    }
  }

  async function ask(req, retries) {
    if (cache.has(req.key)) return cache.get(req.key);
    let r = await once(req);
    while ((!r || !r.ok) && !(r && r.expired) && retries-- > 0) {
      await new Promise((res) => setTimeout(res, RETRY_MS));
      r = await once(req);
    }
    if (r && r.ok && r.status) cache.set(req.key, r);
    return r;
  }

  function refresh(p) {
    const box = ensureBox();
    if (!box) return;
    const req = request(p);
    const key = req ? req.key : null;
    if (key !== current) { box.hidden = true; box.innerHTML = ""; open = false; }
    current = key;
    if (!req) return;
    if (box.hidden) muted(box, "Mislinkage score: scoring…");
    ask(req, 1).then((r) => { if (current === key) paint(box, r, req); });
  }

  document.addEventListener("pyroorb:render", (e) => refresh(e.detail));
})();
