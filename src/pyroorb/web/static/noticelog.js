"use strict";
// PyRoOrb web UI — notice log. Vanilla JS, no build step. Loaded AFTER app.js.
//
// The app pops transient "flash" banners (see flashBanner in app.js). Most hold a
// couple of seconds then fade; ITF-search / linkage banners instead STAY until the
// user dismisses them (their × button, or a click anywhere off the banners). Either
// way, when a banner GOES AWAY a moderate dot its own colour flies up into a corner
// indicator — showing it's being filed — and clicking that indicator opens a
// scrollable log of everything that has flashed.
//
// It hooks in without touching app.js: it pre-creates the #flashHost that
// flashBanner() appends into, then watches that host with a MutationObserver, so
// each banner is classified on the way in and logged + animated on the way out.

(function () {
  const $ = (id) => document.getElementById(id);
  const KIND_COLOR = { info: "#56d4dd", alert: "#f85149", warn: "#f0ad3c" };  // match style.css
  const KIND_LABEL = { info: "Info", alert: "Alert", warn: "Warning" };
  const STICKY_RE = /ITF|linkage/i;   // banners that must not disappear on a timer
  const log = [];                     // { message, kind, ts } — oldest first, filed when a banner leaves

  const escHtml = (s) => String(s).replace(/[&<>]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const escAttr = (s) => escHtml(s).replace(/"/g, "&quot;");

  function kindOf(el) {
    for (const k of Object.keys(KIND_COLOR)) if (el.classList.contains(k)) return k;
    return "info";
  }

  function relTime(ts) {
    const s = Math.round((Date.now() - ts) / 1000);
    if (s < 5) return "just now";
    if (s < 60) return `${s}s ago`;
    const m = Math.round(s / 60);
    if (m < 60) return `${m}m ago`;
    return `${Math.round(m / 60)}h ago`;
  }

  // ---- corner indicator + count badge ------------------------------------
  function refreshIndicator() {
    const btn = $("bannerLogBtn");
    if (!btn) return;
    const dot = btn.querySelector(".log-dot");
    const cnt = $("bannerLogCount");
    if (log.length) {
      const last = log[log.length - 1];
      if (dot) dot.style.background = KIND_COLOR[last.kind] || KIND_COLOR.info;  // tint = latest kind
      if (cnt) { cnt.textContent = String(log.length); cnt.hidden = false; }
      btn.classList.add("has-log");
    } else {
      if (dot) dot.style.background = "";
      if (cnt) { cnt.hidden = true; cnt.textContent = "0"; }
      btn.classList.remove("has-log");
    }
  }

  // ---- the flying dot -----------------------------------------------------
  // A moderate dot the banner's colour, launched from where the banner was and
  // animated quickly into the corner indicator, which pulses on arrival.
  function flyDotFrom(cx, cy, kind) {
    const btn = $("bannerLogBtn");
    if (!btn) return;
    const b = btn.getBoundingClientRect();
    const ex = b.left + b.width / 2, ey = b.top + b.height / 2;
    const color = KIND_COLOR[kind] || KIND_COLOR.info;
    const d = document.createElement("div");
    d.className = "banner-flydot";
    d.style.background = color;
    d.style.boxShadow = `0 0 12px ${color}`;
    d.style.left = cx + "px"; d.style.top = cy + "px";
    d.style.setProperty("--dx", (ex - cx) + "px");
    d.style.setProperty("--dy", (ey - cy) + "px");
    d.addEventListener("animationend", () => {
      d.remove();
      btn.classList.remove("arrived"); void btn.offsetWidth; btn.classList.add("arrived");
    });
    document.body.appendChild(d);
  }

  // ---- the log panel ------------------------------------------------------
  function renderList() {
    const list = $("bannerLogList");
    if (!list) return;
    if (!log.length) { list.innerHTML = `<div class="log-empty">No notices yet.</div>`; return; }
    list.innerHTML = "";
    for (let i = log.length - 1; i >= 0; i--) {          // newest first
      const e = log[i];
      const row = document.createElement("div");
      row.className = "log-row";
      row.innerHTML =
        `<span class="log-row-dot" style="background:${KIND_COLOR[e.kind] || KIND_COLOR.info}" ` +
          `title="${escAttr(KIND_LABEL[e.kind] || e.kind)}"></span>` +
        `<span class="log-row-msg"></span>` +
        `<span class="log-row-time">${escHtml(relTime(e.ts))}</span>`;
      row.querySelector(".log-row-msg").textContent = e.message;
      list.appendChild(row);
    }
  }

  function ensurePanel() {
    let panel = $("bannerLogPanel");
    if (panel) return panel;
    panel = document.createElement("div");
    panel.id = "bannerLogPanel"; panel.className = "bannerlog"; panel.hidden = true;
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Notice log");
    panel.innerHTML =
      `<div class="bannerlog-hd"><span>Notice log</span>` +
        `<button type="button" id="bannerLogClear" class="bannerlog-clear">Clear</button></div>` +
      `<div id="bannerLogList" class="bannerlog-list"></div>`;
    document.body.appendChild(panel);
    panel.addEventListener("click", (ev) => ev.stopPropagation());  // don't self-close
    $("bannerLogClear").addEventListener("click", () => {
      log.length = 0; refreshIndicator(); renderList(); togglePanel(false);   // clearing also closes
    });
    return panel;
  }

  function togglePanel(force) {
    const panel = ensurePanel();
    const btn = $("bannerLogBtn");
    const show = force === undefined ? panel.hidden : force;
    if (show) {
      renderList();
      const r = btn ? btn.getBoundingClientRect()
                    : { bottom: 48, right: window.innerWidth - 10 };
      panel.style.top = (r.bottom + 6) + "px";
      panel.style.right = Math.max(8, window.innerWidth - r.right) + "px";
      panel.hidden = false;
      if (btn) btn.setAttribute("aria-expanded", "true");
    } else {
      panel.hidden = true;
      if (btn) btn.setAttribute("aria-expanded", "false");
    }
  }

  // ---- classify on the way in, file + animate on the way out --------------
  function onBannerIn(el) {
    const kind = kindOf(el);
    const msgEl = el.querySelector(".flash-msg");
    const message = msgEl ? msgEl.textContent : el.textContent.replace(/\s*×\s*$/, "").trim();
    const r = el.getBoundingClientRect();
    // Stash where it is + what it says so we can file it from here when it leaves.
    el.__fly = { cx: r.left + r.width / 2, cy: r.top + r.height / 2, kind, message, done: false };
    if (STICKY_RE.test(message)) el.classList.add("sticky");   // CSS cancels the auto-fade
  }

  function onBannerOut(el) {
    const f = el.__fly;
    if (!f || f.done) return;
    f.done = true;
    log.push({ message: f.message, kind: f.kind, ts: Date.now() });
    refreshIndicator();
    const panel = $("bannerLogPanel");
    if (panel && !panel.hidden) renderList();
    flyDotFrom(f.cx, f.cy, f.kind);
  }

  // ---- wire-up ------------------------------------------------------------
  function init() {
    // Pre-create the flash host so flashBanner() reuses it and we can watch it.
    let host = $("flashHost");
    if (!host) {
      host = document.createElement("div");
      host.id = "flashHost"; host.className = "flashhost";
      document.body.appendChild(host);
    }
    new MutationObserver((muts) => {
      for (const m of muts) {
        for (const n of m.addedNodes)
          if (n.nodeType === 1 && n.classList.contains("flash")) onBannerIn(n);
        for (const n of m.removedNodes)
          if (n.nodeType === 1 && n.classList.contains("flash")) onBannerOut(n);
      }
    }).observe(host, { childList: true });

    const btn = $("bannerLogBtn");
    if (btn) btn.addEventListener("click", (ev) => { ev.stopPropagation(); togglePanel(); });

    // A click that isn't ON a banner "starts interacting elsewhere": dismiss the
    // sticky banners (measuring each first for an accurate fly-dot) and close the log.
    document.addEventListener("click", (ev) => {
      if (ev.target.closest && ev.target.closest(".flash")) return;   // interacting WITH a banner
      const h = $("flashHost");
      if (h) h.querySelectorAll(".flash.sticky").forEach((n) => {
        const r = n.getBoundingClientRect();
        if (n.__fly) { n.__fly.cx = r.left + r.width / 2; n.__fly.cy = r.top + r.height / 2; }
        n.remove();
      });
      const p = $("bannerLogPanel");
      if (p && !p.hidden) togglePanel(false);
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") { const p = $("bannerLogPanel"); if (p && !p.hidden) togglePanel(false); }
    });
    refreshIndicator();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
