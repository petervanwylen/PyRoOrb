// The /memory page: renders GET /api/memory.
//
// Deliberately a standalone module rather than part of app.js -- this page has no
// workspace state, and app.js is already large enough.  Everything here is read-only rendering: the server
// decides what is safe to show (shapes and sizes, never contents; session handles
// truncated to their readable prefix, because a full handle is a credential).

const $ = (id) => document.getElementById(id);
let TIMER = null;

// The whole-heap ("misc") pass walks everything the process can reach and holds
// the GIL while it does, so the server gates it behind $PYROORB_MEMORY_PASSWORD
// (or a loopback caller when none is set) -- see server._misc_allowed. The
// password is held per BROWSER TAB, in sessionStorage: it should survive the
// page's own refresh and the auto-refresh timer, and not outlive the tab on a
// shared machine. It is sent as a header, never in the URL, so it stays out of
// access logs and proxy traces.
const PW_KEY = "pyroorb.memoryPassword";
function pw() {
  try { return sessionStorage.getItem(PW_KEY) || ""; } catch (e) { return ""; }
}
function setPw(v) {
  try {
    if (v) sessionStorage.setItem(PW_KEY, v);
    else sessionStorage.removeItem(PW_KEY);
  } catch (e) { /* private mode: the password just lives for this render */ }
}

function esc(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
const mb = (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(1));

function card(k, v, unit, cls) {
  return `<div class="card ${cls || ""}"><div class="k">${esc(k)}</div>
    <div class="v">${esc(v)}${unit ? ` <small>${esc(unit)}</small>` : ""}</div></div>`;
}

function bar(v, total, cls) {
  const pct = total > 0 ? Math.max(0, Math.min(100, (v / total) * 100)) : 0;
  return `<div class="bar-outer"><div class="bar-inner ${cls || ""}"
    style="width:${pct.toFixed(1)}%"></div></div>`;
}

// --- the headline cards: process, container, and how the total splits ------
function renderCards(d) {
  const p = d.process || {}, c = d.container || {}, t = d.totals || {};
  const out = [];
  out.push(card("Resident (RSS)", mb(p.rss_mb), "MB"));
  out.push(card("Peak RSS ever", mb(p.rss_peak_mb), "MB",
                "" /* VmHWM: what the OOM killer saw */));
  // The headroom the ITF sweep floor consults -- below the floor, sweeps truncate.
  let hcls = "";
  if (c.available_mb !== null && c.available_mb !== undefined && c.sweep_min_free_mb) {
    hcls = c.available_mb < c.sweep_min_free_mb ? "bad"
         : (c.available_mb < c.sweep_min_free_mb * 1.5 ? "warn" : "");
  }
  // the ENFORCED floor (scaled down on a small container); show the configured
  // value alongside only when the two differ, so the scaling is visible
  let floorTxt = "MB";
  if (c.sweep_min_free_mb) {
    floorTxt = `MB · floor ${c.sweep_min_free_mb}`;
    if (c.sweep_min_free_configured_mb &&
        c.sweep_min_free_configured_mb !== c.sweep_min_free_mb) {
      floorTxt += ` (of ${c.sweep_min_free_configured_mb} configured)`;
    }
  }
  out.push(card("Headroom", c.available_mb === null || c.available_mb === undefined
    ? "n/a" : mb(c.available_mb), floorTxt, hcls));
  out.push(card("Tables", mb(t.tables_combined_mb), "MB"));
  if (t.misc_mb !== null && t.misc_mb !== undefined) {
    out.push(card("Misc heap", mb(t.misc_mb), "MB"));
    out.push(card("Native / mapped", mb(t.native_mb), "MB"));
  }
  const s = d.searches || {};
  out.push(card("Searches running", s.running || 0, "", s.running ? "warn" : ""));
  out.push(card("Threads", p.threads === undefined ? "—" : p.threads));
  $("cards").innerHTML = out.join("");
}

// --- one row per table, with its per-entry detail underneath --------------
function renderTables(d) {
  const t = d.totals || {}, total = t.rss_mb || t.accounted_mb || 1;
  const rows = [];
  const loaded = (d.tables || []).filter((x) => x.loaded)
    .sort((a, b) => (b.mb || 0) - (a.mb || 0));
  for (const tab of loaded) {
    const cap = tab.cap ? `${tab.count} / ${tab.cap}`
                        : (tab.count === null || tab.count === undefined ? "" : `${tab.count}`);
    rows.push(`<tr>
      <td><b>${esc(tab.title)}</b>${tab.truncated
          ? ' <span style="color:var(--warn)">(walk truncated)</span>' : ""}
        <div class="note">${esc(tab.note || "")}${tab.detail ? " · " + esc(tab.detail) : ""}</div></td>
      <td class="num">${esc(cap)}</td>
      <td class="num">${mb(tab.mb)}</td>
      <td class="bar-cell">${bar(tab.mb, total)}</td></tr>`);
    for (const r of tab.rows || []) {
      const bits = [r.n_obs ? `${r.n_obs} obs` : "", r.note || ""].filter(Boolean).join(" · ");
      rows.push(`<tr class="sub"><td>${esc(r.id)} <span class="dim">${esc(r.label || "")}</span>
        ${bits ? `<span class="dim"> — ${esc(bits)}</span>` : ""}</td>
        <td class="num"></td><td class="num">${mb(r.mb)}</td><td></td></tr>`);
    }
    if (tab.rows_omitted) {
      rows.push(`<tr class="sub"><td class="dim">… ${tab.rows_omitted} smaller
        ${tab.rows_omitted === 1 ? "entry" : "entries"} not listed</td>
        <td class="num"></td><td class="num"></td><td></td></tr>`);
    }
  }
  const absent = (d.tables || []).filter((x) => !x.loaded)
    .map((x) => esc(x.key) + (x.error ? ` <span style="color:var(--bad)">(${esc(x.error)})</span>` : ""));

  let html = `<h2>Long-lived tables</h2><table>
    <tr><th>Table</th><th class="num">entries</th><th class="num">MB</th><th></th></tr>
    ${rows.join("")}</table>`;
  if (t.overlap_mb > 0.05) {
    html += `<div class="note" style="margin-top:6px">Counted once, held by more than one
      table: <b>${mb(t.overlap_mb)} MB</b> (gross ${mb(t.tables_gross_mb)} MB).
      A search stores its fit in a session <i>and</i> its payload in the jobs table;
      a pasted object sits in a session <i>and</i> in fitcache.</div>`;
  }
  if (absent.length) {
    html += `<div class="note" style="margin-top:6px">Not loaded in this process:
      ${absent.join(", ")} <span class="dim">(never imported to measure them)</span></div>`;
  }
  return html;
}

// --- propagator sweeps: what each thread is integrating right now ----------
function renderSweeps(d) {
  const s = d.sweeps;
  if (!s) return "";
  const orbit = (el, withR) => (!el || el.q_au === undefined || el.q_au === null) ? "—"
    : `q ${esc(el.q_au)} · e ${esc(el.e)}` + (withR ? ` · r ${esc(el.r_au)}` : "");
  const who = (r) => esc(r.backend) + (r.label ? " · " + esc(r.label) : "");
  let html = "";
  if (s.inflight && s.inflight.length) {
    const rows = s.inflight.map((r) => {
      const prog = (r.at === null || r.at === undefined) ? "—" : `${r.at + 1} / ${r.n}`;
      const target = (r.target_days === null || r.target_days === undefined) ? "—"
        : `${esc(r.target_days)} d`;
      return `<tr${r.wedged ? ' class="bad"' : ""}><td class="mono">${esc(r.thread)}</td>
        <td>${who(r)}</td><td class="num">${esc(r.elapsed_s)} s${r.wedged ? " ⚠" : ""}</td>
        <td class="num">${esc(prog)}</td><td class="num">${target}</td>
        <td class="num">${orbit(r.elements, true)}</td></tr>`;
    }).join("");
    html += `<h2>Propagator sweeps in flight</h2><table>
      <tr><th>thread</th><th>backend</th><th class="num">elapsed</th><th class="num">obs</th>
      <th class="num">target</th><th class="num">trial orbit (au)</th></tr>${rows}</table>
      <div class="note" style="margin-top:6px">A sweep older than ${esc(s.wedged_sec)} s is
      flagged: a thread parked inside one integrate() call, which no budget or deadline can
      reach. The JSON carries the full state and epoch, so the case can be re-run offline.</div>`;
  }
  if (s.slow && s.slow.length) {
    const rows = s.slow.map((r) => `<tr><td class="mono">${esc(r.finished)}</td>
      <td>${who(r)}</td><td class="num">${esc(r.elapsed_s)} s</td>
      <td class="num">${r.steps_done === null || r.steps_done === undefined ? "—" : esc(r.steps_done)}</td>
      <td class="num">${esc(r.n)}</td><td class="num">${orbit(r.elements, false)}</td></tr>`).join("");
    html += `<h2>Slowest sweeps lately</h2><table>
      <tr><th>finished (UTC)</th><th>backend</th><th class="num">elapsed</th>
      <th class="num">steps</th><th class="num">obs</th><th class="num">trial orbit (au)</th></tr>
      ${rows}</table>
      <div class="note" style="margin-top:6px">Sweeps slower than ${esc(s.slow_sec)} s, newest
      first — the step count says whether the integrator crawled (millions of steps) or the arc
      was merely long.</div>`;
  }
  return html;
}

// --- running searches: elapsed + how far RSS moved since each started -----
function renderSearches(d) {
  const s = d.searches || {};
  if (!s.rows || !s.rows.length) return "";
  const rows = s.rows.map((r) => `<tr><td class="mono">${esc(r.id)}</td>
    <td class="num">${esc(r.elapsed_s)} s</td>
    <td class="num">${mb(r.rss_start_mb)}</td>
    <td class="num">${r.rss_delta_mb === null || r.rss_delta_mb === undefined
      ? "—" : (r.rss_delta_mb > 0 ? "+" : "") + mb(r.rss_delta_mb)}</td></tr>`).join("");
  return `<h2>Running ITF searches</h2><table>
    <tr><th>job</th><th class="num">elapsed</th><th class="num">RSS at start</th>
    <th class="num">RSS since</th></tr>${rows}</table>
    <div class="note" style="margin-top:6px">RSS movement is a hint, not an attribution —
    anything else running moves it too. The walked cost of a search's own working set is the
    “Working memory of running threads” row above.</div>`;
}

// --- misc: everything no table reaches ------------------------------------
function renderMisc(d) {
  const m = d.misc;
  // Locked is not an error and not "nothing to see": the tables above are the
  // whole report bar this one bucket, so say precisely what is missing and what
  // would unlock it, rather than leaving a blank section that reads as a fault.
  if (!m && d.misc_locked) {
    return `<h2>Misc — locked</h2><div class="locked">
      <b>The whole-heap pass did not run.</b> ${esc(d.misc_locked.message)}
      <div class="note" style="margin-top:6px">It walks every object the process
      can reach, under the GIL, so it is gated to keep an anonymous caller from
      spending the server's CPU on it. Everything above is unaffected.</div></div>`;
  }
  if (!m) return `<h2>Misc</h2><div class="note">Skipped (tables-only mode).</div>`;
  const types = (m.by_type || []).map((r) => `<tr><td class="mono">${esc(r.type)}</td>
    <td class="num">${esc(r.count)}</td><td class="num">${mb(r.mb)}</td>
    <td class="bar-cell">${bar(r.mb, m.mb, "misc")}</td></tr>`).join("");
  const big = (m.biggest || []).map((r) => `<tr><td class="mono">${esc(r.what)}</td>
    <td class="num"></td><td class="num">${mb(r.mb)}</td><td></td></tr>`).join("");
  return `<h2>Misc — heap no table above reaches (${mb(m.mb)} MB, ${m.objects} objects)</h2>
    <table><tr><th>type</th><th class="num">count</th><th class="num">MB</th><th></th></tr>
    ${types}</table>
    ${big ? `<h2>Largest individual objects in misc</h2><table>
      <tr><th>object</th><th class="num"></th><th class="num">MB</th><th></th></tr>${big}</table>`
    : `<div class="note" style="margin-top:6px">No single object ≥ 0.5 MB outside the
       tables — misc is the interpreter and library baseline, spread thin.</div>`}
    ${m.truncated ? `<div class="note" style="color:var(--warn)">Walk hit the node budget;
      misc is a lower bound.</div>` : ""}`;
}

function renderFoot(d) {
  const t = d.totals || {};
  return `Totals: tables <b>${mb(t.tables_combined_mb)}</b> + misc <b>${mb(t.misc_mb)}</b>
    = <b>${mb(t.accounted_mb)} MB</b> of reachable Python objects, against
    <b>${mb(t.rss_mb)} MB</b> resident — leaving <b>${mb(t.native_mb)} MB</b> native:
    the interpreter and its C extensions (NumPy / rebound / ASSIST buffers), the
    memory-mapped DE440 ephemerides, and allocator slack. That part does not shrink by
    trimming any cache on this page.
    <br>Report cost ${esc(d.cost_secs)} s · pid ${esc(d.pid)} ·
    <code>GET /api/memory</code> returns this as JSON (<code>?tables=1</code> skips misc).
    Sizes and shapes only — never contents; handles are truncated.`;
}

// Show the password box only when the server says a password would actually
// unlock this (one is configured and ours is missing or wrong). Where none is
// configured, offering the box would invite the user to guess at a lock that
// has no key -- the message in the misc section says so instead.
function renderLock(d) {
  const locked = d.misc_locked;
  const canAsk = !!(locked && locked.password_required);
  $("pwBox").hidden = !canAsk;
  $("lock").hidden = !(pw() && !canAsk);
  if (locked && locked.reason === "password-wrong") {
    $("err").textContent = locked.message;
  }
}

async function refresh() {
  $("refresh").disabled = true;
  try {
    const headers = {};
    if (pw()) headers["X-Memory-Password"] = pw();
    const r = await fetch("/api/memory" + ($("fast").checked ? "?tables=1" : ""),
                          { headers });
    // A reply WITHOUT our `ok` field is not ours: the hosted proxy answers a
    // server that did not respond (restarting, wedged, out of memory) with its
    // own JSON or HTML.  Say that, with the status, rather than "report failed".
    let d = null;
    try { d = await r.json(); } catch (_) { d = null; }
    if (!d || !("ok" in d)) {
      throw new Error(`the server did not answer (HTTP ${r.status}` +
        (d && d.message ? `: ${d.message}` : "") +
        ") -- the reply came from the proxy in front of it, not from the report");
    }
    if (!d.ok) throw new Error(d.error || `report failed (HTTP ${r.status})`);
    $("err").textContent = "";
    $("when").textContent = d.when + (d.cached_age_s
      ? ` (cached ${d.cached_age_s}s)` : "");
    renderLock(d);
    renderCards(d);
    $("body").innerHTML = renderSweeps(d) + renderSearches(d) + renderTables(d) + renderMisc(d);
    $("foot").innerHTML = renderFoot(d);
  } catch (ex) {
    $("err").textContent = "Could not read /api/memory: " + ex.message;
  } finally {
    $("refresh").disabled = false;
  }
}

function unlock() {
  const v = $("pw").value.trim();
  if (!v) return;
  setPw(v);
  $("pw").value = "";
  refresh();
}

$("refresh").addEventListener("click", refresh);
$("fast").addEventListener("change", refresh);
$("unlock").addEventListener("click", unlock);
$("pw").addEventListener("keydown", (e) => { if (e.key === "Enter") unlock(); });
$("lock").addEventListener("click", () => { setPw(""); refresh(); });
$("auto").addEventListener("change", () => {
  if (TIMER) { clearInterval(TIMER); TIMER = null; }
  if ($("auto").checked) TIMER = setInterval(refresh, 10000);
});
refresh();
