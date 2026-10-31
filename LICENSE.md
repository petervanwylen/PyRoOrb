# License

Copyright (C) 2026 Peter VanWylen and contributors.

This project (the `mvp/` Python orbit-determination code and accompanying files) is licensed
under the **GNU General Public License, version 3.0 or later (GPL-3.0-or-later)**.

```
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

The complete GPL-3.0 text is available at <https://www.gnu.org/licenses/gpl-3.0.txt>
and should be bundled as a `COPYING` file when distributing.

---

## Why GPL-3.0 (compatibility rationale)

The license is **dictated by our runtime dependencies**, not chosen freely. The code
`import`s **REBOUND**, which is **GPL-3.0**. Distributing a combined/derivative work that
links a GPL-3.0 library requires the whole to be offered under GPL-3.0 (or a later GPL).
GPL-3.0 is simultaneously **compatible with every other component we use**, so it is the
correct umbrella for the whole:

| Component | How used | License | GPL-3.0 compatible? |
|---|---|---|---|
| **REBOUND** | imported (runtime) | GPL-3.0 | ✅ (this is what *requires* GPL-3.0) |
| astropy | imported | BSD-3-Clause | ✅ permissive → GPL-compatible |
| scipy | imported | BSD-3-Clause | ✅ |
| numpy | imported | BSD-3-Clause | ✅ |
| astroquery | imported | BSD-3-Clause | ✅ |
| sbpy | imported (designation pack/unpack) | BSD-3-Clause | ✅ |
| jplephem | imported | MIT | ✅ |
| adam_core | install-test only (not in pipeline) | MIT | ✅ |
| **find_orb** | **read only — no code copied** | GPL-2.0 | see note ▼ |
| `ObsCodes` data | redistributed (MPC public data) | MPC public / GPL-2.0 in find_orb tree | ✅ data, see note ▼ |
| **ASSIST** | imported (runtime force model: JPL DE440 + GR + asteroids) | GPL-3.0 | ✅ (same copyleft as REBOUND) |
| JPL DE440 / sb441 ephemeris | data files (downloaded at runtime) | JPL/Caltech public | ✅ data |
| orbitdeterminator / poliastro / hapsira | referenced only — no code taken | GPL-3.0 / MIT | n/a |

See `PROVENANCE.md` for the full read/import/emulate/copy breakdown.

### Note on find_orb (GPL-2.0)

`find_orb` is **GPL-2.0**. GPL-2.0-**only** and GPL-3.0 are *mutually incompatible* — so we are
careful that **no find_orb source code is copied or linked** into this project. find_orb was
**read for design understanding only**; all our code is independently written (see
`PROVENANCE.md`). Because no GPL-2.0 code is incorporated, there is no v2/v3 conflict.

### Note on the `ObsCodes` data file

Our code reads the MPC observatory parallax table. The underlying data is the **Minor Planet
Center**'s public observatory list. The data itself is factual/public; to keep distribution cleanly
independent of find_orb's GPL-2.0 tree, ship a copy fetched directly from the MPC
(<https://www.minorplanetcenter.net/iau/lists/ObsCodes.html>) instead.

---

## Third-party notices

This product uses, but does not include the source of, the imported libraries listed above;
each is governed by its own license, retained by its respective copyright holders. Data obtained
at runtime from **JPL Horizons**, the **JPL Small-Body Database**, and the **Minor Planet Center**
is the property of those providers and subject to their terms.

## If you need a non-copyleft (permissive) license instead

GPL-3.0 here is forced solely by the REBOUND import. To relicense this project permissively
(e.g. BSD/MIT/Apache-2.0), replace REBOUND with a non-GPL propagator — for example a pure
two-body/Kepler propagator, an Apache/BSD N-body integrator, or your own force model — after
which only permissively-licensed dependencies remain and any OSI-approved license becomes viable.
