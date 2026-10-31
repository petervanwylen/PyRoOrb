// Top-bar "Tools" menu (sibling module of app.js -- see noticelog.js for why new UI
// lands in its own file). It only OPENS and CLOSES the dropdown; links that web add-ons
// contribute are placed in the markup by the server. The items inside keep their ids, so app.js and
// randomobj.js drive them exactly as they did in the bar (ITF search enabled once an
// orbit is loaded, ITF trace shown after a search, the random countdown).
//
// Closing: an item that ACTS (a random draw, a family pick, ITF search, ITF trace,
// a link) closes the menu; the random family caret keeps it open, since its
// submenu lives inside. A capture listener sees every click first, so it works even
// though randomobj.js stops propagation on its own buttons.
(function () {
  const btn = document.getElementById("toolsBtn");
  const menu = document.getElementById("toolsMenu");
  if (!btn || !menu) return;                 // markup absent (a stripped test page)

  const setOpen = (open) => {
    menu.hidden = !open;
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    if (!open) {
      const rm = document.getElementById("randomMenu");
      if (rm) rm.hidden = true;
    }
  };

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(menu.hidden);
  });
  menu.addEventListener("click", (e) => {
    const t = e.target.closest("button, a");
    if (!t || t.disabled) return;
    if (t.id === "randomMenuBtn") return;    // opens the family submenu in place
    setTimeout(() => setOpen(false), 0);     // let the item's own handler run first
  }, true);
  menu.addEventListener("click", (e) => e.stopPropagation());   // clicks inside stay inside
  document.addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setOpen(false); });

  // The identification-queue link: its documentation page on the queue's OWN host
  // (IDQUEUE_URL), which links on to the paste/upload page and the dashboard. The
  // href in the markup is only the fallback when the server cannot say.
  const qlink = document.getElementById("idqueueLink");
  if (qlink) {
    fetch("/api/idqueue_links")
      .then((r) => r.json())
      .then((p) => { if (p && p.ok && p.policies) qlink.href = p.policies; })
      .catch(() => { /* keep the fallback */ });
  }
})();
