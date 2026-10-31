"use strict";
// PyRoOrb web UI — final submit confirmation. Vanilla JS, no build step.
// Loaded AFTER app.js; app.js's runItfSubmit() awaits confirmLinkageSubmit().
//
// Submitting an identification to the MPC is the one outward, irreversible
// action in the UI, so the last click-through has to show exactly what is being
// claimed. A window.confirm cannot: it lumped every id in the document into one
// "Tracklets:" list, so the object's OWN packed designation (K25T66D) read as if
// it were one of the ITF tracklets about to be attached. Here the two are kept
// apart -- designations the document names vs tracklets being attached -- and
// the full JSON document is shown verbatim underneath, because the only way to
// be sure of what you are transmitting is to read it.
//
// The client never sends the document: /api/itf_submit takes the session and the
// server rebuilds + re-vets it. What is displayed is the document the server
// built for the observations currently included -- the same input, so the same
// document, modulo the pre-submit DES=DES dedup the server does at send time.

(function () {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  let ov = null, resolveFn = null;

  // Pull the claim out of the document the server built, KEEPING the two kinds
  // of id apart. `designations` are packed ids the MPC already knows (for an ITF
  // find that is the object itself; for a user-assembled identification it is
  // every designation being merged); `trksubs` are the [trksub, date, station]
  // tracklets. Falls back to itf.tracklets only when there is no document.
  function splitClaim(itf) {
    const links = ((itf || {}).linkage || {}).links || {};
    const desigs = [], trks = [];
    for (const key of Object.keys(links)) {
      const l = links[key] || {};
      for (const d of l.designations || []) desigs.push(d);
      for (const t of l.trksubs || []) trks.push(`${t[0]} (${t[1]} ${t[2]})`);
    }
    if (!desigs.length && !trks.length) {
      for (const t of (itf || {}).tracklets || []) trks.push(`${t.trksub} (${t.date} ${t.stn})`);
    }
    return { desigs, trks };
  }

  function build() {
    if (ov) return;
    ov = document.createElement("div");
    ov.id = "submitConfirmOverlay";
    ov.className = "overlay";
    ov.hidden = true;
    ov.innerHTML =
      '<div class="dialog" style="max-width: 820px; width: 92vw;">' +
      '  <div class="dlg-hd"><span id="submitConfirmTitle">Submit identification</span>' +
      '    <small id="submitConfirmSub">this transmits, and cannot be undone</small></div>' +
      '  <div id="submitConfirmBody" class="about-body"></div>' +
      '  <div class="dlg-ft">' +
      '    <div class="spacer"></div>' +
      '    <button id="submitConfirmCopy">Copy JSON</button>' +
      '    <button id="submitConfirmCancel">Cancel</button>' +
      '    <button class="primary" id="submitConfirmGo">Submit to MPC</button>' +
      '  </div>' +
      '</div>';
    document.body.appendChild(ov);
    ov.querySelector("#submitConfirmCancel").addEventListener("click", () => done(false));
    ov.querySelector("#submitConfirmGo").addEventListener("click", () => done(true));
    ov.querySelector("#submitConfirmCopy").addEventListener("click", (e) => {
      const pre = ov.querySelector("#submitConfirmJson");
      navigator.clipboard.writeText(pre ? pre.textContent : "");
      e.target.textContent = "Copied";
      setTimeout(() => { e.target.textContent = "Copy JSON"; }, 1200);
    });
    // clicking off the dialog cancels -- the safe direction
    ov.addEventListener("click", (e) => { if (e.target === ov) done(false); });
    document.addEventListener("keydown", (e) => {
      if (!ov.hidden && e.key === "Escape") done(false);
    });
  }

  function done(ok) {
    if (!ov) return;
    ov.hidden = true;
    const f = resolveFn; resolveFn = null;
    if (f) f(ok);
  }

  // Show the dialog; resolves true if the user confirms, false otherwise.
  // `current` is the CURRENT fit payload, `warns` the quality cautions already
  // clicked through (restated here so the last screen is not silent about them).
  // `route`: "queue" (default -- the PyRoOrb identification queue, which checks,
  // paces and forwards it) or "mpc" (straight to the MPC); `me`: {name, email}.
  // The document as it will be SIGNED: the server holds only a header-less draft
  // (it has no identity of its own), so the preview -- and "Copy JSON" -- carry
  // the person's own header, never an empty one.
  function previewDocument(linkage, me) {
    if (!linkage || !linkage.links) return "(no document on this session)";
    if (!me || !me.name || !me.email) return "(no submitter chosen -- nothing can be sent)";
    return JSON.stringify({ header: { name: me.name, email: me.email, comment: "REPLY_EMAIL" },
                            links: linkage.links }, null, 2);
  }
  window.confirmLinkageSubmit = function (current, warns, route, me) {
    build();
    const toMpc = route === "mpc";
    ov.querySelector("#submitConfirmTitle").textContent = toMpc
      ? "Submit identification directly to the Minor Planet Center"
      : "Submit identification to the PyRoOrb identification queue";
    ov.querySelector("#submitConfirmSub").textContent = toMpc
      ? "this transmits to the MPC, skipping the queue, and cannot be undone"
      : "the queue checks it, adds any ITF tracklets, and forwards it to the MPC";
    ov.querySelector("#submitConfirmGo").textContent = toMpc ? "Submit to MPC" : "Submit to queue";
    const itf = (current || {}).itf || {};
    const ident = itf.kind === "ident";
    const who = ident
      ? ((current.designations || [current.object]).join(" = "))
      : ((current.designations && current.designations[0]) || current.object);
    const { desigs, trks } = splitClaim(itf);
    warns = warns || [];

    // ids are monospace (they are ids); the prose cautions are not
    const list = (items, cls) => `<ul class="claim${cls ? " " + cls : ""}">` +
      items.map((x) => `<li>${esc(x)}</li>`).join("") + "</ul>";
    let html = `<p><b>Object:</b> ${esc(who)}` +
      (itf.neocp ? ' <span class="chip sat">NEOCP</span>' : "") + "</p>";
    if (me) {
      html += `<p><b>Submitted as:</b> ${esc(me.name)} &lt;${esc(me.email)}&gt; ` +
        '<span class="about-sub">(change it from the ▾ menu)</span></p>';
    }
    if (ident) {
      html += "<p>Claiming these are <b>ONE object</b>:</p>" + list(desigs.concat(trks));
    } else {
      if (desigs.length) {
        html += "<p>Designation(s) the document names " +
          "(the object itself &mdash; <i>not</i> a tracklet being added):</p>" + list(desigs);
      }
      html += `<p>ITF tracklet(s) being attached to it (${trks.length}):</p>` +
        (trks.length ? list(trks) : "<p class=\"about-sub\">none</p>");
    }
    if (warns.length) {
      html += `<p style="color:var(--warn)">&#9888; ${warns.length} quality ` +
        "caution(s) were flagged and acknowledged:</p>" + list(warns, "prose");
    }
    html += "<p><b>Document to be submitted</b> <span class=\"about-sub\">" +
      "(built by the server from the observations currently included; it re-builds " +
      "and re-vets it at send time, and may collapse designations the MPC has " +
      "already identified with each other)</span></p>" +
      "<pre id=\"submitConfirmJson\">" + esc(previewDocument(itf.linkage, me)) + "</pre>";
    ov.querySelector("#submitConfirmBody").innerHTML = html;
    ov.hidden = false;
    return new Promise((res) => { resolveFn = res; });
  };
})();
