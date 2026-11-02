# PyRoOrb web UI — Phase 0 (read-only viewer)

The first runnable slice of the web UI: fetch/fit an
object and **see** it — elements, residual table, a live 3D orbit, and the O−C
residual plot. No editing yet (that's Phase 1).

## Run it

```bash
.venv/bin/python -m pyroorb.web.devserver     # auto-reloading dev runner (recommended)
# or, no auto-reload:
.venv/bin/pyroorb-web        # binds 0.0.0.0:8000
# then open it in a browser and hit "Fit" (defaults to 2008 TE4)
```

`--port N` / `--host H` to change the bind.

**`devserver.py`** runs `server.py` as a child and restarts it whenever any
`.py` under `src/pyroorb/` changes, so backend edits go live without a manual restart
(static html/css/js never needed one — just refresh). `data` folders,
`__pycache__` and `tests` are ignored, so normal fitting doesn't trigger a
reload; a syntax error just stops the child until the next save fixes it.

### Reaching it from your host browser (microVM / sandbox dev)

When the server runs inside a Linux microVM/sandbox and your browser is on the
macOS host, you need **no Python/conda on the host** — only a browser. All deps
stay in the VM; the host just renders HTML. The server defaults to
`--host 0.0.0.0`, which is **required** for port forwarding (a service bound to
`127.0.0.1` inside the sandbox is not reachable through a published port).

**Docker Sandbox (`docker sandbox run claude`).** Ports are *not* published by a
flag on `run`; you publish them after the sandbox is up. On the **host**:

```bash
docker sandbox ls                                  # find the sandbox name
docker sandbox ports <name> --publish 8000:8000    # host:sandbox
# then open http://localhost:8000   (unpublish later with --unpublish 8000:8000)
```

(`sbx ports <name> --publish 8000:8000` is the same if the shorthand is on PATH.)

**Other setups.** If the host can route to the VM subnet, `http://<VM-IP>:8000`
works directly (find it with `ip -4 addr` inside the VM, e.g. `192.168.65.3`).
An SSH tunnel also works where available: `ssh -L 8000:localhost:8000 <vm>`,
then `http://localhost:8000`. **Zero dependencies** — pure Python
stdlib backend + vanilla-JS front end, so it runs offline with no `pip install`
and no Node build step. It fits **cache-first**: any object with a
`ades_<desig>.json` file in the cache directory fits with no network (while it is
fresh); uncached objects need network and will error cleanly if offline.

For a **combined linkage**, enter comma-separated designations
(e.g. `2008 TE4, 2008 UF`).

## The central pane (adding one, or a button to one)

The middle column is a single panel whose title is the `#paneSelect` dropdown
(Observations & residuals / Orbarium / View Ephemeris / Earth Impacts). Each pane owns
**exactly two elements**, both tagged with the same `data-pane` key as its `<option>`:

- a control strip in the panel header — `<span class="panectl" data-pane="ephem">` —
  holding every button that pane, and only that pane, provides;
- a body — `<div class="bd panebody" data-pane="ephem">` — holding its content.

`setCentralPane()` (app.js) shows the matching pair and hides every other by that same
rule, from the `CENTRAL_PANES` registry, which also carries the pane's `onShow` hook and
any side panel it owns elsewhere in the layout (`aux`; the on-sky panel belongs to the
ephemeris). Adding a pane = one `<option>`, one strip, one body, one registry row. A
button inside a strip may still hide itself on app state — that is the pane's own
business, and it can no longer leak onto anything else.

Two rules keep it that way, both asserted by `tests/test_pane_controls.js`:

1. **Never put an inline `display` on a `.panectl`/`.panebody`.** An inline style
   outranks any stylesheet rule — that is exactly how `#obsControls`' `display:flex`
   kept the observation buttons (⤓ worst, Full step, MPC 80, ＋ addl info …) on screen
   over the ephemeris, orbarium and impact panes. The CSS guard is
   `.panectl[hidden], .panebody[hidden] { display: none !important }`.
2. **A pane-owned element whose renderer also decides its visibility** (the "NOT FINAL
   ORBIT" replay banner) is tagged `class="paneowned" data-pane="…"` and toggled with
   `paneStateShow(id, on)` rather than `.hidden` — the framework contributes the
   "is my pane showing" half of the answer and re-applies it on every switch.

## What you get (Phase 0)

| Panel | Source | Notes |
|---|---|---|
| Status chip (RMS, used/total, arc, opp, ok/suspect, PHA) | `FittedOrbit.as_dict()` | |
| Orbital elements (a,e,q,Q,i,Ω,ω,M,Tp,P ± 1σ) | `as_dict()` | |
| Physical/class (class, MOID, T_J, U, H, epoch, force model) | `as_dict()` | |
| Non-gravitational model (SRP 1/2/3-param, comet A1/A2/A3, Yarkovsky) — menu mirrors find_orb's force-model list | `refine_nongrav` via `/api/nongrav` | dropdown re-solves the orbit with out-gassing/Yarkovsky/SRP terms; shows the fitted A ± 1σ, detection σ, and the before→after RMS. ASSIST backend only |
| Observations & residuals table | `fit.residuals()` + parsed obs | sort worst-first; resid sparkbar; excluded rows red. Virtualized above 1500 rows (only the viewport window is in the DOM, re-rendered on scroll) so mega-objects show every obs without a DOM blow-up; ≤1500 renders in full |
| Impact location (central-pane dropdown) | `impact.find_impact` via `/api/impact` | Phase-1 impact monitoring: propagates the fit forward and shows where/when it reaches the ground (100 km atmospheric entry + ground point, geodetic lat/lon with a map link, entry speed & slope) — or the closest-approach miss distance if it doesn't. Nominal trajectory only (no footprint yet); validated to ~km on 2008 TC3 |
| Impact **probability** (experimental; below the location card) | `impact.impact_probability` via `/api/impact_probability` | Phase-2: LOV b-plane impact probability on demand (optional encounter-date target / window). Carries a **loud red warning** — research prototype, defers to JPL Sentry & ESA NEODyS. Validated vs Sentry (2013 TP4); ultra-short arcs only order-correct |
| Observatory roll-up (counts, median residual/code) | derived | |
| 3D heliocentric orbit | Kepler sampler in `engine_api` | drag to rotate, scroll to zoom; Earth/Mars/Jupiter reference circles |
| O−C vs time plot | `fit.residuals()` | RA·cosδ (orange) / Dec (blue), included vs excluded |
| Fit-stage history | `fit.history` | subarc → extend → reject → redeem … |

## Files

- `engine_api.py` — **framework-agnostic** bridge: `run_fit(desig)` → one
  JSON-serialisable dict. No web imports, so the planned FastAPI app reuses it
  verbatim. Also holds the elements→xyz Kepler sampler for the 3D view
  (visualisation only; no new science).
- `server.py` — stdlib `http.server`; serves the static SPA + `/api/fit`,
  `/api/nongrav` (+ `/api/nongrav_models`), `/api/fullstep`, `/api/ephem`,
  `/api/impact`, `/api/random_families` + `/api/random_object`, exports.
  A request whose `Host` is exactly a name in `_RETIRED_HOSTS` (the old
  `web-production-fee0b.up.railway.app`) is permanently redirected to the same path on
  `www.pyroorb.net` (GET 301, POST 308); every other host is served normally.
- `addons.py` — **web add-ons.** The server knows no optional feature by name: an add-on
  (a `WebAddon`) registers its own GET/POST routes, path prefixes, pages and static files
  (served after the server's own `/static/` files), Tools-menu links, and the scripts and
  stylesheets `index.html` loads (placed at the `<!-- web add-ons: ... -->` markers), plus a
  start hook for background work. Add-ons are found through the `pyroorb.web_addons`
  entry-point group and through `web_addons.txt` at the checkout root (one
  `module:attribute` per line; `PYROORB_WEB_ADDONS_FILE` overrides the path). A broken
  add-on is logged and skipped. Handlers get a `Context` (the bounded worker pool, the
  background ITF slot, the station table, a displayed fit).
- `fitqueue.py` — **admission control for fits.** A `ThreadingHTTPServer` puts no
  bound on concurrency, so a hundred people curious about one object used to be a
  hundred simultaneous fits: the container hits its memory limit and is OOM-killed
  (and `railway.json` stops restarting after three), while any slow fit outlives
  the hosted proxy's request limit. So a web fit is now submitted to a bounded
  worker pool sized from the container's OWN cgroup CPU and memory limits, and the
  browser polls for it. Three properties matter:
  * **Bounded** — the failure mode becomes *slow*, not *fatal*. Past the queue depth
    a request is refused in plain words instead of accepted and never reached.
  * **Single-flight** — work is keyed by CONTENT (`fit_key`: designations + force +
    engine revision), so identical requests attach to the fit already running
    rather than each starting their own. A cache only helps *after* the first fit;
    the herd that matters arrives before it. A finished job stays keyed for
    `DONE_TTL`, which makes this the in-memory L1 response cache too.
  * **Private sessions** — a shared fit hands back a `restore` token, never the
    interactive `session` handle, because one session mutated by a hundred people
    is a hundred people editing each other's orbit. `engine_api.session_from_job`
    clones a private session from the job's pristine template by rebuilding the
    evaluator (`fit_object._build_evaluator`) — measured at ~4 ms against a whole
    fit, because finding the state is the expensive part and rebuilding the
    machinery around it is not. That also makes session expiry RECOVERABLE for the
    first time: the job outlives the session table, so an evicted session is
    re-claimed instead of re-fitted.
  Surfaces: `POST /api/fit_start` → `{job, waiting, ahead, eta_sec}`, polled from
  `GET /api/fit_result?id=`; `POST /api/session {job}` claims a private session;
  `GET /healthz` reports queue depth, worker occupancy and RSS against the
  container limit (the Railway healthcheck points at `/`, which keeps serving
  static HTML while every fit times out, so it can never say anything is wrong).
  `GET /api/fit` still answers inline for a quick object and hands back a job
  handle for a slow one — it goes through the same pool, because a route that
  skipped it would make the pool decorative. Env: `PYROORB_FIT_WORKERS`,
  `PYROORB_FIT_QUEUE_MAX`, `PYROORB_FIT_PEAK_MB`, `PYROORB_SESSION_MAX`.
- `fitdisk.py` — the **ephemeral** on-disk tier under `fitqueue`. A CAPACITY
  store, not a durable one: it lives on container scratch (`PYROORB_FIT_CACHE_DIR`,
  default `$TMPDIR/pyroorb-fitcache`) and is *meant* to vanish on restart or
  redeploy, because a new deployment may compute a different orbit from the same
  arc and a cache that outlived the process would serve answers the running code
  never produced. It REFUSES to sit under a persistent volume (`/data` and
  friends) rather than quietly filling it. What it buys is reach (a scratch disk
  holds thousands of fits where RAM holds dozens, so far more distinct objects
  stay warm through a spike) and relief (a payload on disk is a payload not in
  the process). Only the serialised payload is stored — the live template
  session is not, since it holds a rebound simulation — so a disk hit serves the
  ~96% who look and leave at full speed and carries `refit` instead of a
  `restore` token; the few who then interact pay one fresh fit. Write-through
  happens after the waiters are released, so a slow disk never delays a fit.
  LRU by mtime under `PYROORB_FIT_CACHE_MB` (256), stops writing below
  `PYROORB_FIT_CACHE_MIN_FREE_MB` (512) because a full disk breaks far more than
  a cache miss does, and a torn file is deleted and treated as a miss.
- `ratelimit.py` — per-client limits on the routes that START WORK. The queue
  bounds what the box will *do*; this bounds *whose* work gets done, and neither
  substitutes for the other: a fair queue filled by one script in a loop tells
  every real visitor the server is busy while behaving exactly as designed.
  Token buckets per (client, kind) — `fit` 12 burst / 12·min⁻¹, `paste` 6, `itf`
  3 burst / ~1·min⁻¹, `random` 10 — each tunable by env
  (`PYROORB_RL_*_BURST` / `_RATE`, `PYROORB_RL_DISABLED` to switch it off). The
  POLLING routes (`/api/fit_result`, `/api/itf_result`, `/api/itf_trace`) are
  deliberately exempt and must stay that way: a client polls them *because* the
  work is slow. Client identity comes from the first `X-Forwarded-For` hop
  (behind a proxy the peer is the proxy, so everyone would share one bucket) —
  which is forgeable, so this is a courtesy limit against runaway scripts, NOT a
  security control; real abuse protection belongs at the edge. The client table
  is bounded and idle-evicted, or the limiter becomes the leak it prevents.
  `server._MAX_BODY_BYTES` (8 MB) caps request bodies: the paste routes used to
  read `Content-Length` bytes with no ceiling, so one POST could ask the process
  to allocate anything it liked, before any queue could see it.
- `static/fitjob.js` — the client half: starts the job, polls it, and renders the
  wait as "2 fits ahead of you, about 40 s" rather than an unexplained spinner.
  Also owns `ensureSession`, which claims (or re-claims) a private session.
- `static/randomobj.js` — the top bar's 🎲 **Random** split button: draws a real
  object out of the cached MPCORB index (`engine_api.random_object`, capped at
  `RANDOM_MAX_OPPOSITIONS`=7 so one click can't launch a multi-thousand-observation
  megafit) and fits it; the caret picks the family (NEO / MBA / Mars crosser / Hilda /
  Trojan / Centaur / TNO / high-inclination), served by `/api/random_families` so the
  menu can't drift from the engine's own definitions. One draw per 30 s, stamped in
  `localStorage` so a reload isn't a bypass.
- `itf_wording.py` — plain English for the ITF search's internal shorthands, on
  everything the browser is shown (banners, warnings, the live trace, the
  vetting block). The engine counts oppositions fractionally (1.5 = two
  oppositions with only one of them multi-night, 1.75 = three with only one)
  and reports an outcome as 0 / 0.6 / 1 so code can compare them; a reader gets
  "2 oppositions, one of them a single night" and "possible linkage (not
  submittable)" instead. Presentation only — `itf/` and the CLI keep the
  numbers.
- **Stopping an ITF search** — `static/itfcancel.js`: while a search runs, "stop
  search" sits next to "see trace" and the trace viewer shows "Stop search". Both
  `POST /api/itf_cancel {job}` (`engine_api.itf_search_cancel`), which sets the job's
  cancel token and answers at once. Python cannot kill a thread, so the stop is
  cooperative (`deadline.cancellable` / `deadline.Cancelled`): the search unwinds at
  its next propagator step, database batch or between-fit budget check, and the next
  `/api/itf_result` poll answers `{ok: false, cancelled: true}` (shown as "stopped",
  never "failed"). The admission slot is released only once the worker has actually
  unwound, since until then it still holds the memory the slot accounts for.
  `Cancelled` is deliberately NOT a `DeadlineExceeded`: the search's per-trial-fit
  handler catches that by name, and a cancel caught there would only skip one trial.
  The job id is the handle the page chose, so only the page that started a search can
  stop it; the NEOCP monitor's background slots carry no token and cannot be stopped
  from the web.
- `static/` — `index.html`, `app.js` (render + canvas 3D + SVG plot), `style.css`.

## Not in Phase 0 (see the plan for the roadmap)

Editing/toggling obs + re-fit (Phase 1), ephemeris generator + sky view
(Phase 1), teaching trace/step-through (Phases 2–3), uncertainty cloud (Phase 4).
The eventual stack is React+TS+Three.js (plan §8); this vanilla build is the
zero-friction Phase-0 prototype that proves the data path end-to-end.
