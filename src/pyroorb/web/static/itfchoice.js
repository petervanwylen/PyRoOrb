"use strict";
// PyRoOrb web UI — choose between two ITF search results. Vanilla JS, no build step.
// Loaded AFTER app.js; app.js's runItfSearch() awaits chooseItfResult(p) when the
// search answered with two results.
//
// An ITF search can end holding two different answers: a linkage it would submit,
// and a fuller unconfirmed ("possible", 0.6) one that links more of the ITF -- the
// extra oppositions too thin to confirm (a lone night or two). The server builds a
// whole fit for each and sends the second as `itf.alternative`; this dialog asks
// which one to show. Most searches return one answer and never see it.

(function () {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  let ov = null, resolveFn = null, choices = null;

  function build() {
    if (ov) return;
    ov = document.createElement("div");
    ov.id = "itfChoiceOverlay";
    ov.className = "overlay";
    ov.hidden = true;
    ov.innerHTML =
      '<div class="dialog" style="max-width: 760px; width: 92vw;">' +
      '  <div class="dlg-hd"><span>The ITF search found two results</span>' +
      '    <small>which one should be shown?</small></div>' +
      '  <div id="itfChoiceBody" class="about-body"></div>' +
      '  <div class="dlg-ft">' +
      '    <div class="spacer"></div>' +
      '    <button id="itfChoiceSubmittable">Submittable version</button>' +
      '    <button class="primary" id="itfChoiceComplete">Most complete version</button>' +
      '  </div>' +
      '</div>';
    document.body.appendChild(ov);
    ov.querySelector("#itfChoiceSubmittable").addEventListener("click", () => done("submittable"));
    ov.querySelector("#itfChoiceComplete").addEventListener("click", () => done("complete"));
    // dismissing takes the submittable one -- the result the search stands behind
    ov.addEventListener("click", (e) => { if (e.target === ov) done("submittable"); });
    document.addEventListener("keydown", (e) => {
      if (!ov.hidden && e.key === "Escape") done("submittable");
    });
  }

  function done(which) {
    if (!ov) return;
    ov.hidden = true;
    const f = resolveFn, c = choices;
    resolveFn = null; choices = null;
    if (f) f(c[which]);
  }

  function describe(p, title, note) {
    const itf = p.itf || {};
    const trks = (itf.tracklets || []).map((t) =>
      `<li>${esc(t.trksub)} (${esc(t.date)} ${esc(t.stn)}, ${t.n_obs} obs)</li>`).join("");
    return `<p><b>${title}</b> &mdash; ${(itf.tracklets || []).length} tracklet(s), ` +
      `+${itf.added_obs || 0} obs. <span class="about-sub">${note}</span></p>` +
      `<ul class="claim">${trks}</ul>`;
  }

  // Resolves to the payload to show: `p` itself or `p.itf.alternative`, with the
  // pairing stripped from both so the page holds one ordinary result.
  window.chooseItfResult = function (p) {
    const alt = p && p.itf && p.itf.alternative;
    if (!alt || !alt.itf) return Promise.resolve(p);
    delete p.itf.alternative;
    if (alt.itf) delete alt.itf.alternative;
    const [sub, full] = p.itf.submittable ? [p, alt] : [alt, p];
    build();
    choices = { submittable: sub, complete: full };
    ov.querySelector("#itfChoiceBody").innerHTML =
      describe(sub, "Submittable version",
               "the linkage the search confirms; it can be submitted to the MPC") +
      describe(full, "Most complete version",
               "links more of the ITF, but the extra observations are too thin to " +
               "confirm: shown for review, never submittable") +
      '<p class="about-sub">Closing this dialog shows the submittable version.</p>';
    ov.hidden = false;
    return new Promise((res) => { resolveFn = res; });
  };
})();
