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

## Schedules (`.github/workflows/`)

| Workflow | Cadence | Job |
|---|---|---|
| `daily-refresh.yml` | every 6h + nightly ET | full refresh → gates → bundle → build check → commit+push → Vercel rebuild |
| `live-refresh.yml` | every 10 min | `--live`: while a tracked event is on, re-read its pages (1–3 requests), rebuild, push |
| `validate.yml` | every 30 min | independent freshness check (Liquipedia vs repo vs deployed site); restarts live refresh when behind; fails red |

Concurrency: refresh workflows share one queue (`daily-refresh` group) — two runs
never write the cache together.

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
