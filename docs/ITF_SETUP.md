# Running ITF searches against your own SBN database

**Most people should not need this.** ITF searches can be run on **[pyroorb.net](https://www.pyroorb.net)**
(the web UI's **ITF search**), which keeps its own index current. This guide is for running
`pyroorb-itf` yourself.

Two ways to run it yourself:

- **Use the author's database** — ask first (a GitHub issue on this repository); access is by
  permission.
- **Your own clone of the SBN database** — the setup below.

> **Before any mass ITF searching or submitting, please give the author a polite heads-up**
> (a GitHub issue). The author already runs these searches across the ITF and submits the
> identifications they find. Several people running the same code over the same tracklets
> produce the same identifications several times over, and every duplicate is work the Minor
> Planet Center has to process and discard.

## What the search needs

The search reads the SBN replica of the MPC database (PostgreSQL) and a small index that
PyRoOrb builds and keeps current inside it.

| | |
|---|---|
| **Server** | `SBN_DB_HOST` (port 5432, database `sbn`) |
| **Read-only search role** | `pub`, password in `SBN_PUB_PASSWORD` — reads `obs_sbn`, `mpc_orbits`, `current_identifications`, `numbered_identifications`, `primary_objects` |
| **Index role** | `itf`, password in `ITF_DB_PASSWORD` — owns the `itf` schema the maintenance jobs write |

Set them in the environment, or in the `.env` settings file (see
[INSTALL.md](INSTALL.md#settings-one-env-file) for where it may live):

```bash
SBN_DB_HOST=your-sbn-replica.example.org
SBN_PUB_PASSWORD=...
ITF_DB_PASSWORD=...
```

Install PyRoOrb with the ITF extra: [`install.sh`](../install.sh) includes it by default (and
on macOS installs Homebrew's `libomp`, which the ranker's xgboost needs); see
[INSTALL.md](INSTALL.md). By hand: `uv pip install "pyroorb[itf] @ git+https://github.com/petervanwylen/PyRoOrb"`
(on macOS also `brew install libomp`).

## The tracklet index

`itf.itf_tracklets` is a one-row-per-tracklet index — on-sky motion with its uncertainty,
HEALPix tile, mean magnitude — over the population the search cares about:

- **ITF tracklets**: every observation unattributed (`provid IS NULL AND status = 'I'` in
  `obs_sbn`);
- **pseudo-ITF**: designation families with at most three tracklets, or with no orbit in
  `mpc_orbits`.

The search queries this index (about 3 million rows) one HEALPix window at a time, instead of
scanning `obs_sbn` (over 500 million rows). Every eviction is logged in
`itf.itf_tracklets_evictions`.

### One-time setup

```bash
pyroorb-itf maintain ensure-schema    # create the itf tables and indexes (idempotent)
pyroorb-itf maintain backfill         # build the index from obs_sbn (long; resumable)
pyroorb-itf maintain status           # row counts and the sync watermark
```

The backfill is checkpointed in the database: if it is interrupted, run it again and it
continues.

### Keeping it current

`obs_sbn` changes all the time — new tracklets arrive and identifications attribute ITF
tracklets to known objects — so the index must be maintained. Three jobs, all idempotent:

| Job | Cadence | What it does |
|---|---|---|
| `refresh` | every 1–10 minutes | admits new ITF tracklets, refreshes changed ones, and removes those no longer valid (identified, mixed, or gone) |
| `reconcile` | daily | full pass over the pseudo-ITF designation families (tens of minutes); needs a direct, long-lived database connection |
| `sweep` | weekly | chunked full-table backstop for removals that leave no incremental trace |

Run them from your own scheduler (`pyroorb-itf maintain refresh`, and so on), or let PyRoOrb
run all three on one clock:

```bash
ITF_MAINTAIN_TURNED_ON=True pyroorb-itf maintain scheduler   # forever: refresh ~60 s, daily, weekly
pyroorb-itf maintain tick                                    # one cycle and exit (for an external cron)
```

`pyroorb-web` runs the same scheduler as a background thread when `ITF_MAINTAIN_TURNED_ON=True`.

### Checking it is healthy

```bash
pyroorb-itf maintain events    # each job's state and the recent event log
pyroorb-itf maintain status    # row counts, watermark, evictions
```

The scheduler records every failure and every daily or weekly success in
`itf.itf_maintain_events`, so maintenance health can be checked from anywhere with the `itf`
credentials.

## Searching

```bash
pyroorb-itf search "2017 CL37"              # the live decision trace prints as it runs
pyroorb-itf search "2017 CL37" --thorough   # sweep fainter (slower)
```

Searches are dry runs: nothing is sent to the MPC unless you pass `--submit` and have set
`ITF_SUBMIT_NAME` / `ITF_SUBMIT_EMAIL` (the document's header). Please read the note at the
top before submitting at any scale.
