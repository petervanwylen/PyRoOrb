// The sky check page: a field query, or the identification of pasted astrometry
// (src/pyroorb/skycheck/webaddon.py serves the API). Long checks come back as a
// pending job stub and are settled through FitJob (fitjob.js).
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    try { return JSON.parse(await r.text()); }
    catch (e) { return { ok: false, error: "the server did not answer readably (" + r.status + ")" }; }
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  const num = (v, d) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));

  function sexa(raDeg, decDeg) {
    let h = ((raDeg % 360) + 360) % 360 / 15;
    const hh = Math.floor(h), mm = Math.floor((h - hh) * 60);
    const ss = ((h - hh) * 60 - mm) * 60;
    const sign = decDeg < 0 ? "-" : "+";
    const d = Math.abs(decDeg), dd = Math.floor(d), dm = Math.floor((d - dd) * 60);
    const ds = ((d - dd) * 60 - dm) * 60;
    const p = (x, w) => String(x).padStart(w, "0");
    return [`${p(hh, 2)} ${p(mm, 2)} ${ss.toFixed(2).padStart(5, "0")}`,
            `${sign}${p(dd, 2)} ${p(dm, 2)} ${ds.toFixed(1).padStart(4, "0")}`];
  }

  function kindText(o) {
    if (o.variants) return `NEOCP (${o.in_field}/${o.variants} variants, spread ${num(o.spread, 0)}")`;
    if (o.kind === "asteroid" && o.U != null) return `asteroid U=${o.U}`;
    return o.kind;
  }

  function objectTable(objs, tracklet) {
    const head = ["Designation", "RA", "Dec",
      tracklet ? "offset\"" : "sep\"", "dRA\"", "dDec\"",
      ...(tracklet ? ["motion rms\""] : []),
      "mag", "RA \"/min", "Dec \"/min", "r", "Δ", "elong", "kind"];
    const rows = objs.map((o) => {
      const [ra, dec] = sexa(o.ra, o.dec);
      const name = o.name && o.name !== o.designation ? `${esc(o.designation)} <span class="status">${esc(o.name)}</span>` : esc(o.designation);
      return `<tr class="${o.match ? "match" : ""}"><td class="l">${name}</td><td>${ra}</td><td>${dec}</td>` +
        `<td>${num(tracklet ? o.mean_offset : o.separation, 1)}</td><td>${num(o.offset_ra, 1)}</td><td>${num(o.offset_dec, 1)}</td>` +
        (tracklet ? `<td>${num(o.motion_rms, 2)}</td>` : "") +
        `<td>${num(o.mag, 1)}</td><td>${num(o.rate_ra, 2)}</td><td>${num(o.rate_dec, 2)}</td>` +
        `<td>${num(o.r, 3)}</td><td>${num(o.delta, 3)}</td><td>${num(o.elongation, 1)}</td>` +
        `<td class="l">${esc(kindText(o))}</td></tr>`;
    });
    return `<div class="tablewrap"><table class="sc"><thead><tr>` +
      head.map((h, i) => `<th class="${i === 0 || i === head.length - 1 ? "l" : ""}">${h}</th>`).join("") +
      `</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
  }

  function renderField(res) {
    const [ra, dec] = sexa(res.ra, res.dec);
    let html = `<div class="panel"><b>${res.objects.length}</b> object(s) within ${res.radius_arcmin}' of ` +
      `${ra} ${dec} at JD ${res.time_utc.toFixed(5)} UTC, station ${esc(res.obscode)}`;
    (res.notes || []).forEach((n) => { html += `<div class="notes">note: ${esc(n)}</div>`; });
    if (res.objects.length) html += objectTable(res.objects, false);
    $("results").innerHTML = html + "</div>";
  }

  function renderIdentify(res) {
    const n = res.tracklets.length;
    const nid = res.tracklets.filter((t) => t.identification).length;
    const nref = res.tracklets.filter((t) => t.refused).length;
    let html = `<div class="panel"><b>${n}</b> tracklet(s) from ${res.n_obs} observation(s): ` +
      `<b>${nid}</b> identified` + (nref ? `, <b>${nref}</b> not checked` : "") + `</div>`;
    res.tracklets.forEach((t) => {
      const [ra, dec] = sexa(t.ra, t.dec);
      const verdict = t.refused ? `<span class="none">not checked</span>`
        : t.identification ? `<span class="id">= ${esc(t.identification)}</span>`
        : `<span class="none">${t.ambiguous ? "ambiguous" : "no match"}</span>`;
      html += `<div class="panel tracklet"><h3>${esc(t.label || "?")} · ${esc(t.obscode)} · ` +
        `${t.n_obs} obs · ${ra} ${dec} · ${verdict}</h3>` +
        (t.refused ? `<div class="notes">${esc(t.refused)}</div>`
         : t.candidates.length ? objectTable(t.candidates, true)
         : `<div class="notes">no known object predicted near it</div>`) +
        `</div>`;
    });
    $("results").innerHTML = html;
  }

  async function run(url, body, statusEl, render) {
    statusEl.textContent = "checking…";
    statusEl.classList.remove("error");
    let res = await postJSON(url, body);
    if (window.FitJob && window.FitJob.isPending(res)) {
      res = await window.FitJob.settle(res, (t) => { statusEl.textContent = t; });
      if (res && res.result) res = res.result;
    }
    if (!res || !res.ok) {
      statusEl.textContent = (res && res.error) || "the check failed";
      statusEl.classList.add("error");
      return;
    }
    statusEl.textContent = "";
    render(res);
  }

  $("fGo").addEventListener("click", () => run("/api/skycheck/field", {
    ra: $("fRa").value.trim(), dec: $("fDec").value.trim(),
    time: $("fTime").value.trim() || null, station: $("fStation").value.trim() || "500",
    radius: $("fRadius").value.trim(), mag_limit: $("fMag").value.trim(),
    comets: $("fComets").checked, neocp: $("fNeocp").checked,
  }, $("fStatus"), renderField));

  $("iGo").addEventListener("click", () => run("/api/skycheck/identify", {
    text: $("iText").value, radius: $("iRadius").value.trim(), match: $("iMatch").value.trim(),
    motion: $("iMotion").value.trim(), comets: $("iComets").checked,
    neocp: $("iNeocp").checked,
  }, $("iStatus"), renderIdentify));

  function tab(which) {
    $("tabField").classList.toggle("on", which === "field");
    $("tabIdentify").classList.toggle("on", which === "identify");
    $("paneField").classList.toggle("hidden", which !== "field");
    $("paneIdentify").classList.toggle("hidden", which !== "identify");
    $("results").innerHTML = "";
  }
  $("tabField").addEventListener("click", () => tab("field"));
  $("tabIdentify").addEventListener("click", () => tab("identify"));

  fetch("/api/skycheck/status").then((r) => r.json()).then((s) => {
    const d = s.defaults || {};
    // the engine's defaults, never literals in the page (empty means "the default" anyway)
    [["fRadius", d.radius], ["iRadius", d.radius], ["iMatch", d.match], ["iMotion", d.motion]]
      .forEach(([id, v]) => { if (v != null && !$(id).value) $(id).value = v; });
    if (!s.ready) { $("catInfo").textContent = "the orbit catalogue is being prepared (first start: a few minutes)"; return; }
    const built = s.built ? new Date(s.built * 1000).toISOString().slice(0, 16).replace("T", " ") : "?";
    let info = `${(s.orbits || 0).toLocaleString()} orbits (MPCORB + comets), ` +
      `catalogue built ${built} UTC`;
    if (s.max_days && s.epoch_jd) {
      // the server's limit as calendar dates (JD 2440587.5 = 1970-01-01 0h UTC)
      const day = (jd) => new Date((jd - 2440587.5) * 86400000).toISOString().slice(0, 10);
      info += `; times from ${day(s.epoch_jd - s.max_days)} to ${day(s.epoch_jd + s.max_days)} ` +
        `(within ${s.max_days} days of the catalogue's epoch)`;
    }
    if (s.config_error) info += `; configuration error: ${s.config_error}`;
    $("catInfo").textContent = info;
  }).catch(() => {});
})();
