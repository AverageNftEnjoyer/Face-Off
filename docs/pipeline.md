# Faceoff data pipeline

Fan-analytics CS2 prediction hub. One principle governs everything below:
**`data/matches.json` is the only thing ingested; every other number is a pure
function of it, recomputed at build time.**

## Data flow

```
Liquipedia (MediaWiki API)          Valve regional_standings (git)
        |                                          |
        v                                          v
data/lpfetch.py ── polite cached fetch ──► data/vrs.py ──► data/vrs.json
(2.5s gap, retries w/ backoff,                       (read-only pull, date logged)
 STALE_SERVED ledger when throttled)
        |
        v
scripts/daily_refresh.py ── tournament discovery ──► data/raw/lp_titles_selected.txt
(tier categories + series listings;                  (tracked S/A-tier pages)
 live mode: --live refreshes only pages
 of in-progress events)
        |
        v
data/collect_liquipedia.py ── parse {{Match}} ──► data/matches.json  (FULL REBUILD
templates, de-dupe by (date, teams,              every run: re-running is inherently
hltv id), finished series only                   idempotent) + data/team_aliases.json
                                                 (output only, never read back)
        |
        +── data/rosters.py ──► data/roster_events.json (stand-ins; SUPERVISED:
        |                       candidate staged, review via a roster-review issue, then
        |                       `python scripts/promote_rosters.py --yes`)
        +── viewer/fetch_assets.py ──► viewer/assets.json (logos, map art)
        +── data/lineups.py ──► data/lineups.json (incremental; failure-tolerant,
                                never breaks the refresh)
        |
        v
scripts/data_gates.py ── quality gates on the rebuilt data ──► BLOCKS the publish
(schema, duplicate IDs, sane counts,                     on failure and opens a
 team-name resolution, Elo-delta bounds)                 GitHub issue with the report
        |
        v (gates pass)
viewer/build_bundle.py ──► viewer/data-bundle.json   (versioned; see
(recomputes EVERYTHING from                        docs/bundle-schema.md;
 matches.json: Elo, form, map pools,               GITIGNORED: rebuilt by CI
 H2H, predictions, veto sims,                      and by Vercel's buildCommand
 track record)                                     on every deploy)
        |
        v
viewer/build_viewer.py ──► static HTML (bundle inlined: DATA + FIXTURES +
(renders EXCLUSIVELY from        RESULTS + META) ──► Vercel builds on push
 the bundle; fails loud
 on missing/corrupt/
 foreign-version bundle)
```

## SQLite copy for querying (`data/faceoff.db`)

`scripts/build_db.py` rebuilds `data/faceoff.db` from `data/*.json` after the
gates pass, in both refresh workflows, and it is committed with the data. It is
a read-only convenience copy: nothing in the pipeline reads it, and
`matches.json` stays the source of truth. Tables: `matches`, `maps`,
`elo_history` (point-in-time series Elo before / after every series), `teams`,
`lineups`, `standins`, `vrs_standings`, `lower_matches`, `lower_maps`; views
`team_matches` (one row per team per series) and `team_map_stats`. Every table
and column is described in `schema_doc` (`SELECT * FROM schema_doc`). The file
is replaced only when its bytes change, so an unchanged input never commits.

## Schedules (`.github/workflows/`)

| Workflow | Cadence | Job |
|---|---|---|
| `daily-refresh.yml` | every 6h + nightly ET | full refresh → gates → bundle → build check → commit+push → Vercel rebuild |
| `live-refresh.yml` | every 10 min | `--live`: while a tracked event is on, re-read its pages (1–3 requests), rebuild, push |
| `validate.yml` | every 30 min | independent freshness check (Liquipedia vs repo vs deployed site); restarts live refresh when behind; fails red |

Concurrency: refresh workflows share one queue (`daily-refresh` group) — two runs
never write the cache together.

## Veto distribution cache

`data/veto_dist_cache.json` (committed; Vercel's clean checkout reads it) caches
the per-matchup pick / ban chances that `viewer/build_viewer.py::run_vdist`
otherwise enumerates (about 80% of a bundle build, ~13,500 veto trees). A warm
build only recomputes matchups whose veto inputs changed.

- **Key**: sha1 of the canonical JSON of exactly the fields
  `predictor.veto_distribution` reads: `map_pool, maps_a, maps_b, veto_ev_a,
  veto_ev_b, veto_format, permaban_a, permaban_b, veto_prefix` (whole values,
  floats by repr). Ratings, h2h, form, rosters and names are not read by the
  veto, so a rating-only change is a hit.
- **Fingerprint** (file header): sha1 of `veto.py` (line endings normalised),
  the source of the predictor functions on the veto path, every `CONFIG`
  `veto_*` / `map_*` entry, `VETO_ORDER`, the VD formats and the cache layout.
  Any difference discards the whole file, so a model or CONFIG change can never
  serve a stale answer.
- **Pruning / writes**: after each build only the entries used by that build
  are kept; the file is written atomically (temp file + replace), one entry per
  line with sorted keys so daily diffs stay small. A corrupt, foreign-version or
  other-fingerprint file is ignored with a one-line warning, never fatal.
- **Output is identical** cached or not (the file stores chances in thousandths,
  the precision the page encoding keeps).
- `FACEOFF_NO_VD_CACHE=1` disables reading and writing. `FACEOFF_VD_CACHE_FILE`
  points a build at another cache file (tests, experiments).
- The last line of `run_vdist` reports inputs, hits, computed and seconds.
- The cache is refreshed by whatever run builds the bundle and commits `data/`
  (the daily refresh, or a developer build); a build that only renders the
  committed data never needs to recompute.

## Failure philosophy

- Silent on success, LOUD on failure. A green run must mean the live data is current.
- `data/refresh_status.json` records pages Liquipedia refused; the workflow fails
  instead of shipping a green run with old results.
- Gate failures block the publish AND open a GitHub issue with the gate report —
  never a silent red X, never a bad publish.
- Fallback: if a fresh bundle fails, the site keeps serving the last good bundle
  with its age labeled. Empty/broken is the only unforgivable state.

## Known hard parts

- **Rosters** have no structured feed. Changes are staged as candidates and only
  applied after human confirm — auto-applying them corrupts everything downstream.
- **Liquipedia** lags hours after matches and its HTML drifts. Parsing is isolated
  behind one function; structural parse failures fail the job loudly.
- **HLTV** has no official API; it is not scraped. Liquipedia lag is accepted and
  labeled via `data_through`.

## When Liquipedia refuses the runner

GitHub's hosted runners share IP addresses, and Liquipedia rate-limits by address, so a
request can be refused (HTTP 429) even at one request per run.

- A **live** run (`daily_refresh.py --live`, every 10 minutes) makes one short retry and
  stops. It never waits in a loop: the next scheduled run is the retry. (An earlier
  version waited out three rounds of backoff and held the shared queue for 22 minutes.)
- A **daily** run waits once (2 minutes) and retries, then reports.
- Pages that stayed stale are written to `data/refresh_status.json` with `stale_since`,
  the time the outage started. The file only changes when an outage starts or ends, so
  it does not commit on every run.
- `daily_refresh.py --check-status --max-stale-minutes N` is the last workflow step. The
  live workflow uses 30: a single refusal is only a warning, and the run turns red once
  the pages have been stale for 30 minutes. The daily workflow uses 0 (fail at once).
- Backstop while a tournament is on and GitHub is being refused: run
  `python scripts/daily_refresh.py --live` from your own machine (a home address is not
  rate-limited the same way), then commit and push the changed `data/` files.
