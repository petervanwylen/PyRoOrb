// Top-bar "Random asteroid" button (sibling module of app.js -- see the note in
// noticelog.js: app.js is at its practical size limit, so new UI features land in
// their own file and reach into app.js's globals, which are script-scoped and shared).
//
// Click the button -> GET /api/random_object?family=<selected> -> drop the designation
// into the search box and run the normal fit. The caret opens a family menu (NEO / MBA
// / TNO / Trojan / ...), built from GET /api/random_families so the choices and the
// opposition cap come from the engine rather than from literals here (the ITF-modal
// lesson: a UI literal silently shadows the engine default when the engine moves).
//
// The button is rate-limited to one draw per RANDOM_COOLDOWN_MS. That is partly
// courtesy to the server -- every draw is a real fetch + fit -- and partly the point:
// a 30-second pause is an invitation to actually look at the object you got rather
// than to reroll through twenty of them.
(function () {
  const RANDOM_COOLDOWN_MS = 30000;
  const LAST_KEY = "pyroorb.randomLastDraw";   // survives a reload -> F5 is not a bypass
  const FAM_KEY = "pyroorb.randomFamily";

  const btn = $("randomBtn");
  const caret = $("randomMenuBtn");
  const menu = $("randomMenu");
  if (!btn || !caret || !menu) return;         // markup absent (a stripped test page)

  let families = [{ key: "any", label: "Any asteroid", hint: "" }];
  let maxOpps = null;
  let family = "any";
  let cooldownTimer = null;

  const store = {                              // localStorage is absent/blocked in places
    get(k) { try { return window.localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { window.localStorage.setItem(k, v); } catch { /* ignore */ } },
  };

  const famLabel = (key) => (families.find((f) => f.key === key) || {}).label || key;

  // ---- rate limit --------------------------------------------------------
  const msLeft = () => {
    const last = Number(store.get(LAST_KEY) || 0);
    if (!Number.isFinite(last) || last <= 0) return 0;
    const left = last + RANDOM_COOLDOWN_MS - Date.now();
    // A clock that jumped backwards (or a tampered value) must not lock the button
    // out for hours -- treat anything beyond the cooldown as expired.
    return left > 0 && left <= RANDOM_COOLDOWN_MS ? left : 0;
  };

  function paint() {
    const left = msLeft();
    const name = family === "any" ? "" : " " + famLabel(family).replace(/ \(.*\)$/, "");
    if (left > 0) {
      btn.disabled = caret.disabled = true;
      btn.textContent = `🎲 ${Math.ceil(left / 1000)}s`;
      btn.title = "One random object every 30 seconds — take a look at this one "
        + "while you wait.";
    } else {
      btn.disabled = caret.disabled = false;
      btn.textContent = `🎲 Random${name}`;
      btn.title = `Fit a random real asteroid${name ? " —" + name : ""}`
        + (maxOpps ? ` (at most ${maxOpps} oppositions, so it fits quickly)` : "");
    }
  }

  // Tick the countdown to zero. Kept separate from stamping the clock so a reload
  // mid-cooldown resumes the REMAINING time instead of restarting a fresh 30 s.
  function runCountdown() {
    if (cooldownTimer) clearInterval(cooldownTimer);
    cooldownTimer = setInterval(() => {
      paint();
      if (msLeft() <= 0) { clearInterval(cooldownTimer); cooldownTimer = null; }
    }, 250);
    paint();
  }

  function startCooldown() {
    store.set(LAST_KEY, String(Date.now()));
    runCountdown();
  }

  // ---- family menu -------------------------------------------------------
  function buildMenu() {
    menu.textContent = "";
    for (const f of families) {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.family = f.key;
      b.textContent = (f.key === family ? "✓ " : "  ") + f.label;
      if (f.hint) b.title = f.hint;
      menu.appendChild(b);
    }
  }

  async function loadFamilies() {
    try {
      const r = await fetch("/api/random_families");
      const p = await r.json();
      if (p && p.ok && Array.isArray(p.families) && p.families.length) {
        families = p.families;
        maxOpps = p.max_oppositions || null;
      }
    } catch { /* keep the one-entry fallback; the draw itself still works */ }
    const saved = store.get(FAM_KEY);
    if (saved && families.some((f) => f.key === saved)) family = saved;
    buildMenu();
    paint();
  }

  // ---- draw --------------------------------------------------------------
  async function draw() {
    if (msLeft() > 0) return;
    btn.disabled = caret.disabled = true;
    btn.textContent = "🎲 …";
    let p;
    try {
      const r = await fetch(`/api/random_object?family=${encodeURIComponent(family)}`);
      p = await r.json();
    } catch (e) {
      p = { ok: false, error: e.message };
    }
    if (!p || !p.ok) {
      // No draw happened, so don't spend the cooldown on it.
      paint();
      flashBanner("Could not pick a random object: " + ((p && p.error) || "request failed"),
                  "alert");
      return;
    }
    startCooldown();                 // spend it only on a real object
    if (typeof gaEvent === "function") gaEvent("get_random_object", { family });
    const bits = [`opposition${p.oppositions === 1 ? "" : "s"} ${p.oppositions}`];
    if (p.a != null) bits.push(`a ${Number(p.a).toFixed(2)} au`);
    if (p.e != null) bits.push(`e ${Number(p.e).toFixed(2)}`);
    if (p.incl != null) bits.push(`i ${Number(p.incl).toFixed(1)}°`);
    if (p.H != null) bits.push(`H ${Number(p.H).toFixed(1)}`);
    flashBanner(`Random ${p.family_label}: ${p.designation} — ${bits.join(", ")}`);
    $("desig").value = p.designation;
    runFit();
  }

  btn.addEventListener("click", (e) => { e.stopPropagation(); draw(); });
  caret.addEventListener("click", (e) => {
    e.stopPropagation();
    if (caret.disabled) return;
    const open = menu.hidden;
    menu.hidden = !open;
    caret.setAttribute("aria-expanded", open ? "true" : "false");
  });
  document.addEventListener("click", () => {
    menu.hidden = true;
    caret.setAttribute("aria-expanded", "false");
  });
  menu.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-family]");
    if (!b) return;
    e.stopPropagation();
    family = b.dataset.family;
    store.set(FAM_KEY, family);
    menu.hidden = true;
    caret.setAttribute("aria-expanded", "false");
    buildMenu();
    paint();
    draw();                          // picking a family is itself a request for one
  });

  paint();
  if (msLeft() > 0) runCountdown();  // a reload mid-cooldown keeps counting down
  loadFamilies();
})();
