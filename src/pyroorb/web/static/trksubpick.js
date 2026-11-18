"use strict";
// PyRoOrb web UI — pick the tracklet a typed trksub means. Vanilla JS, no build step.
// Loaded AFTER app.js; app.js's runFit() hands a failed fit carrying `trksub_question`
// to TrksubPick.show().
//
// A token in the search box that is not a published designation is looked up as a
// submitter's trksub in the SBN replica (obs_sources._fetch_temporary), so an isolated
// tracklet can be linked by typing "H005311 K16FN7V".  A trksub is the submitter's own
// label and is not unique: when one names several tracklets the server does not guess,
// it answers with the list, and this asks the user for ONE of them or ALL of them.  A
// tracklet that already carries a designation is no longer a trksub's business: it is
// said ("H005311 is already 2016 FV237") with a button that types the designation in.
//
// The choice is a rewrite of the search box -- the token replaced by qualified ids
// ("H005311@568/2015-01-26") that name exactly one tracklet each -- followed by a fresh
// fit, so the result is a plain fit of what the box now says and is shareable as a link.

(function () {
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // The box text with `token` replaced by `repl` (an array of ids), every other typed id kept.
  function rewrite(typed, token, repl) {
    const out = [];
    for (const t of typed || []) {
      if (t === token) out.push(...repl);
      else out.push(t);
    }
    return out.join(", ");
  }

  function show(p, box, refit) {
    const q = p.trksub_question || {};
    const typed = p.typed || Object.keys(q);
    let html = "";
    for (const tok of Object.keys(q)) {
      const a = q[tok] || {};
      const des = a.designated || [];
      const ch = a.choices || [];
      html += `<div class="trksub-pick" data-token="${esc(tok)}">`;
      if (des.length) {
        html += des.map((t) =>
          `<p><b>${esc(tok)}</b> (${esc(t.stn)}, ${esc(t.date)}, ${esc(t.n_obs)} obs) is already ` +
          `<b>${esc(t.designation)}</b>. ` +
          `<button type="button" data-use="${esc(t.designation)}">Use ${esc(t.designation)}</button></p>`
        ).join("");
      }
      const nDes = a.n_designated || des.length;
      if (nDes > des.length) {
        html += `<p style="color:var(--muted)">…and ${nDes - des.length} more designated ` +
          `tracklet${nDes - des.length === 1 ? "" : "s"} carry this trksub.</p>`;
      }
      const nCh = a.n_choices || ch.length;
      if (ch.length) {
        html += `<p><b>${esc(tok)}</b> ${nCh > 1
          ? `names ${nCh} undesignated tracklets — which one do you mean?`
          : "also names this undesignated tracklet:"}</p><ul class="claim">` +
          ch.map((c) =>
            `<li><button type="button" data-pick="${esc(c.token)}">${esc(c.stn)} · ` +
            `${esc(c.date)} · ${esc(c.n_obs)} obs</button> ` +
            `<span style="color:var(--muted)">${esc(c.token)}</span></li>`).join("") +
          "</ul>";
        if (nCh > ch.length) {
          // "All" over a list the page did not show would fit tracklets nobody saw.
          html += `<p style="color:var(--muted)">…and ${nCh - ch.length} more: add ` +
            `@station to the trksub (e.g. ${esc(tok)}@${esc(ch[0].stn)}) to narrow the list.</p>`;
        } else if (ch.length > 1) {
          html += `<p><button type="button" data-all="1">All ${ch.length} tracklets</button></p>`;
        }
      }
      html += "</div>";
    }
    box.innerHTML = html || esc(p.error || "");
    box.querySelectorAll(".trksub-pick").forEach((div) => {
      const tok = div.getAttribute("data-token");
      const ch = (q[tok] && q[tok].choices) || [];
      div.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
        let repl;
        if (b.hasAttribute("data-use")) repl = [b.getAttribute("data-use")];
        else if (b.hasAttribute("data-all")) repl = ch.map((c) => c.token);
        else repl = [b.getAttribute("data-pick")];
        refit(rewrite(typed, tok, repl));
      }));
    });
  }

  window.TrksubPick = { show, rewrite };
})();
