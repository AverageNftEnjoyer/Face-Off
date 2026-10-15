# Architecture

Face-Off is a fan-analytics site for CS2: win probabilities, map veto
simulation and calibrated confidence for every tracked S/A-tier series. It is
stdlib-only Python plus one static HTML page, rebuilt by GitHub Actions and
served by Vercel.

One rule governs the design: **`data/matches.json` is the only thing ingested
as results; every number on the site is a pure function of the committed data,
recomputed at build time.**

> **Keep this file current.** Any change that adds, moves, renames or removes a
> module, a data file, a workflow step or a folder updates the matching section
> here **and** adds a line to the [change log](#change-log) in the same commit.
> `tests/test_architecture.py` fails when a code file is missing from the
> module map, when the map names a file that does not exist, or when code
> appears outside the code folders.

---

## Repository layout

```
Face-Off/
├── paths.py            the folder layout in code; every entry point imports it first
├── model/              the prediction engine and the rating / feature history it reads
├── evaluation/         walk-forward evaluation, weight fitting, leakage proofs
├── collectors/         Liquipedia and Valve fetchers + parsers  ->  data/*.json
├── pipeline/           scheduled refresh jobs and data maintenance
├── checks/             data-quality gates and freshness / consistency checks
├── viewer/             bundle + static page builder, page template, site media
├── api/                Vercel serverless functions: the on-demand notes endpoint
├── tests/              unittest suite (python -m unittest discover -s tests)
├── data/               inputs, caches and committed outputs: data only, no code
├── assets/             source art (full-size originals; the site uses viewer/media/)
├── .github/workflows/  daily refresh, live refresh, freshness validation
└── vercel.json         site build: bundle -> page -> status feed, served from public/
```

Folders hold code by **role**, not by layer of an import graph: `model/` may be
imported by everything, `pipeline/` and `checks/` are the scripts the workflows
run, and `viewer/` turns the data into the site.

### Imports: flat module names via `paths.py`

The folders are not Python packages. Each script and test starts with:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (puts every code folder on sys.path)
```

`paths.py` puts every folder in `paths.CODE_DIRS` on `sys.path`, so modules
import each other by flat name (`import backtest`, `import lpfetch`,
`import events`) wherever they live, and any script runs directly as
`python <folder>/<script>.py` from the repo root. It also exports `ROOT`,
`DATA` (`data/`), `RAW` (`data/raw/`) and `VIEWER`; code that reads or writes
data uses `paths.DATA` rather than a path relative to its own file.

Module names must stay unique across all code folders (they share one
namespace). A new code folder goes in `paths.CODE_DIRS`, `pyrightconfig.json`
and the layout above.

---

## Data flow

```
 Liquipedia (MediaWiki API)                 Valve regional_standings (GitHub)
          │                                              │
          ▼                                              ▼
 collectors/lpfetch.py  polite cached fetch     collectors/vrs.py ─────────► data/vrs.json
 (2.5 s gap, retries, stale ledger)             collectors/lineups.py ─────► data/lineups.json
          │                                     collectors/valve_results.py► data/valve_results.json
          ▼                                              (second source, cross-check only)
 pipeline/daily_refresh.py   discovery ─► data/raw/lp_titles_selected.txt
 (--live: only running events)
          │
          ▼
 collectors/collect_liquipedia.py ─► data/matches.json   (full rebuild every run)
                                     data/team_aliases.json
          ├─ collectors/rosters.py ─► roster candidate ─► roster-review issue
          │                           └─ pipeline/promote_rosters.py --yes ─► data/roster_events.json
          ├─ collectors/event_lineups.py ─► data/event_lineups.json
          ├─ viewer/fetch_assets.py ─► viewer/assets.json, data/raw/media/
          └─ pipeline/lower_refresh.py ─► data/matches_lower.json   (last step, daily only)
          │
          ▼
 checks/data_gates.py   schema, duplicates, counts, names, Elo bounds, cross-source
          │             FAIL ─► publish blocked + GitHub issue with the report
          ▼ pass
 pipeline/build_db.py ─► data/faceoff.db   (read-only SQLite copy, committed)
          │
          ▼  commit data/ + push  ─►  Vercel build (vercel.json)
 viewer/build_bundle.py ─► viewer/data-bundle.json  (Elo, form, pools, H2H,
          │                                          predictions, veto, track record,
          │                                          notes: viewer/notes_data.py + model/notes.py
          │                                          over data/notes_facts.json, display only)
          ▼
 viewer/build_viewer.py ─► public/index.html + public/img/   (renders only from the bundle)
          │                 + public/notes/inputs-<v>.json   (static copy of the notes snapshot)
          ▼
 api/notes_api.py  (Vercel function) ◄─ viewer/notes-inputs.json (snapshot, written by build_bundle)
          ▲   GET /api/notes_api?v=..&a=..&b=..   notes computed on request, CDN-cached per URL
          └── the page's notes panels
 viewer/write_status.py ─► public/status.json                (result alerts feed)
```

---

## Module map

Every code file, by folder. (Checked by `tests/test_architecture.py`.)

### Root

| File | Role |
|---|---|
| `paths.py` | Folder layout constants (`ROOT`, `DATA`, `RAW`, `VIEWER`, `CODE_DIRS`); puts the code folders on `sys.path`. |

### `model/`: prediction engine

| File | Role |
|---|---|
| `model/predictor.py` | The engine: a log-odds blend of factors (skill, form, H2H, maps/veto, roster, stakes) with `CONFIG` holding every tunable weight. `predict_match(input)` is the entry point. |
| `model/veto.py` | Veto subsystem: an exact probability distribution over legal map vetoes (`CONFIG["veto_mode"]`). |
| `model/backtest.py` | Point-in-time history: loads `matches.json` / `matches_lower.json` / roster events, folds series Elo day by day (`Timeline`, `build_dataset`), and runs the train/test backtest. Shared by the viewer, gates and evaluation. |
| `model/lineup_features.py` | Point-in-time lineup features from `data/lineups.json` + manual `data/roster_overrides.json`. |
| `model/roster_flags.py` | Display-only "treat with caution" roster flags shown beside predictions (not a model input). |
| `model/tenure.py` | Point-in-time roster tenure (days with the current five, join dates) from dated lineups + squad histories. Display only: tenure failed its held-out test, so it is not a model input; the notes read it. |
| `model/notes.py` | Matchup notes engine and tool: deterministic rules (rest, schedule, tenure, streak, head-to-head, form, stand-ins, rumours) over the bundle's `notes` section; `get_notes` / `get_notes_batch` / `team_notes` + CLI (`docs/notes-tool-manifest.json`). Pure, no I/O; never read by the predictor. |
| `model/notes_facts.py` | Fact schema v1 of the notes' web facts: validation, ids, merge and point-in-time selection (shared by the fetcher and the engine). |

### `evaluation/`: is the model honest?

| File | Role |
|---|---|
| `evaluation/walkforward.py` | Walk-forward fit and evaluation of factor weights; writes `data/walkforward_report.json` (the site's track record). |
| `evaluation/fit_weights.py` | Fits factor multipliers on the train split. |
| `evaluation/leakage_check.py` | Proves every feature is strictly pre-match (scramble / truncate tests with canaries); test 8 covers the matchup notes. |
| `evaluation/notes_eval.py` | Fire rate of each matchup-notes rule over the last 12 months and its threshold sweep (informational, not a signal test); writes `data/notes_eval_report.json`. |

### `collectors/`: sources to `data/`

| File | Role |
|---|---|
| `collectors/lpfetch.py` | Polite, cached Liquipedia API fetcher (`data/raw/pages/`), throttling and stale-page ledger. |
| `collectors/collect_liquipedia.py` | Parses cached tournament wikitext into `data/matches.json`, `team_aliases.json`, `matches_lower.json`. |
| `collectors/rosters.py` | Stand-in / missing-IGL roster events from team pages (supervised: builds a candidate). |
| `collectors/event_lineups.py` | Per-team, per-event lineups from cached tournament pages (no network). |
| `collectors/lineups.py` | Per-match five-man lineups from Valve's regional-standings repository. |
| `collectors/valve_results.py` | Second, independent results source (Valve rows) used only as a cross-check. |
| `collectors/vrs.py` | Valve Regional Standings: `data/vrs.json`, `data/vrs_status.json`; `--check` fails when stale. |
| `collectors/notes_fetch.py` | Web facts for the matchup notes (player form, stand-ins, roster rumours) for teams playing in the next 3 days: one OpenAI Responses API call per team (`web_search`, form + news in one strict JSON answer), at most once per 24 h; validated with `notes_facts`, appended to `data/notes_facts.json`; per-run manifest + log; call cap and a hard monthly budget (`FACEOFF_NOTES_BUDGET_USD`, summed from the log); `--dry-run` on recorded fixtures (`tests/fixtures/notes_fetch/`). Display context only. |
| `collectors/notes_summarize.py` | AI-written match writing: post-match recaps for matches finished in the last 3 days (house style: cause first, map by map, the model check, what's next; figures only from the match data, clichés rejected) and "why the model picks X" summaries for scheduled matches, from the model's own numbers + the notes (no web search); grounding check rejects any number not in the input; rewrites only on changed input; own monthly budget (`FACEOFF_SUMMARY_BUDGET_USD`, default $5). Writes `data/notes_summaries.json`. |

### `pipeline/`: scheduled jobs

| File | Role |
|---|---|
| `pipeline/daily_refresh.py` | The refresh the workflows run: discovery, page refresh (`--live` for running events), rebuild, rosters, assets, VRS, lineups, second source, lower tier; `--check-status` turns long outages red. |
| `pipeline/lower_refresh.py` | Daily refresh of lower-tier results (rating data only; failure only warns). |
| `pipeline/backfill_lower.py` | Slow, resumable one-off backfill of 2025 lower-tier results. |
| `pipeline/roster_watch.py` | Diffs the roster candidate against the live roster file for the review issue. |
| `pipeline/promote_rosters.py` | Rebuilds and promotes the reviewed roster candidate (`--yes`). |
| `pipeline/build_db.py` | Rebuilds `data/faceoff.db` (read-only SQLite copy with a `schema_doc` table). |

### `checks/`: quality and freshness

| File | Role |
|---|---|
| `checks/data_gates.py` | Gates a rebuilt `matches.json` against the published one; blocks the publish on failure. |
| `checks/cross_source_check.py` | Agreement report between `matches.json` and `valve_results.json`. |
| `checks/validate_freshness.py` | Independent schedule: Liquipedia vs repo vs deployed site; restarts the live refresh when behind. |
| `checks/roster_overrides_check.py` | Validates `data/roster_overrides.json` names against known players and teams. |

### `viewer/`: the site

| File | Role |
|---|---|
| `viewer/build_bundle.py` | Emits `viewer/data-bundle.json`, the single versioned derived-data bundle. |
| `viewer/notes_data.py` | The bundle's `notes` section = the notes snapshot: engine inputs from `data/`, cleaned web facts, freshness, next fixtures, `inputs_version`; `write_inputs` writes `viewer/notes-inputs.json`; `page_payload` is the page's small `NOTES` (no notes: the page asks `api/notes_api.py`). |
| `viewer/build_viewer.py` | Computes everything from the data (`build_data`) and renders the static page from `template.html` + the bundle. |
| `viewer/events.py` | Tournament discovery and parsing from cached pages (infobox, format, brackets, schedule). |
| `viewer/fetch_assets.py` | Logos, map art, flags and team-page roster data from Liquipedia into `viewer/assets.json`. |
| `viewer/team_colors.py` | Team colours from logo pixels (stdlib PNG decoder). |
| `viewer/veto_cache.py` | Content-addressed cache of veto distributions (`data/veto_dist_cache.json`), fingerprinted on model code. |
| `viewer/write_status.py` | Writes `public/status.json`, the small feed the page polls for new results. |

Non-code viewer files: `viewer/template.html` (the whole front end: HTML, CSS,
JS), `viewer/assets.json` (asset index, committed), `viewer/media/site_bg.jpg`
(site background).

### `api/`: serverless functions (Vercel)

| File | Role |
|---|---|
| `api/notes_api.py` | `GET /api/notes_api`: matchup / team notes on request from the snapshot `viewer/notes-inputs.json` via `model/notes.py`; structured errors, version pinning (409), CDN cache headers, in-instance LRU; `--serve` runs it locally. Decision record: `docs/notes-contracts.md` section 4. |

### `tests/`

One `test_<area>.py` per area, `unittest` only. `test_architecture.py` keeps
this document and the layout in step.

---

## Data files (`data/`)

| File | Written by | Notes |
|---|---|---|
| `matches.json` | `collect_liquipedia` | Source of truth for S/A results. |
| `matches_lower.json` | `collect_liquipedia` via `lower_refresh` | Lower-tier rating data, never shown as events. |
| `team_aliases.json` | `collect_liquipedia` | Output only. |
| `roster_events.json` | `promote_rosters` | Human-confirmed; never auto-applied. |
| `roster_overrides.json` | by hand | Dated manual lineup moves (`checks/roster_overrides_check.py`). |
| `inactive_teams.json` | by hand | Teams hidden as inactive on the site. |
| `lineups.json` | `collectors/lineups.py` | Incremental; failure-tolerant. |
| `event_lineups.json` | `collectors/event_lineups.py` | |
| `valve_results.json` | `collectors/valve_results.py` | Cross-check only. |
| `vrs.json`, `vrs_status.json` | `collectors/vrs.py` | |
| `refresh_status.json` | `daily_refresh` | Pages Liquipedia refused, with `stale_since`. |
| `veto_dist_cache.json` | `viewer/build_viewer.py` | Build cache, committed so Vercel builds stay warm. |
| `backtest_report.json`, `walkforward_report.json` | `backtest`, `walkforward` | Read by the viewer's track record. |
| `veto_fixtures.json` | by hand | Test fixtures for the veto engine. |
| `faceoff.db` | `build_db` | Read-only SQLite copy. |
| `notes_eval_report.json` | `evaluation/notes_eval.py` | Notes rule fire rates + threshold sweeps (informational). |
| `notes_facts.json` | `collectors/notes_fetch.py` | Web facts for the matchup notes (fact schema v1, append-only, 400-day prune). Never read by the model. |
| `notes_manifest.json` | `collectors/notes_fetch.py` | Latest fetch run: scope, calls, tokens, estimated cost, monthly budget, rejected facts, failures, the `checked` ledger (24 h TTL). |
| `notes_manifest_log.jsonl` | `collectors/notes_fetch.py` | One compact summary line per fetch run (history; the monthly budget is summed from it). |
| `notes_summaries.json` | `collectors/notes_summarize.py` | AI "why" summaries (`summaries`) and post-match recaps (`recaps`), by event match id (kept 120 days). Display only. |
| `notes_summary_log.jsonl` | `collectors/notes_summarize.py` | One line per summary run: calls, tokens, cost, rejections (the budget is summed from it). |
| `SOURCES.md` | by hand | Where every record comes from, and the terms followed. |
| `raw/pages/`, `raw/media/`, `raw/*.txt` | `lpfetch`, `fetch_assets`, `daily_refresh` | Fetch cache, images, tracked page titles. |

Generated and gitignored: `viewer/data-bundle.json`, `viewer/notes-inputs.json`, `public/`, roster and
fill-in candidates, research reports.

---

## Automation

| Workflow | Cadence | Does |
|---|---|---|
| `daily-refresh.yml` | every 6 h + nightly | `pipeline/daily_refresh.py` → `checks/data_gates.py` → bundle → tests → page build check → `pipeline/build_db.py` → commit + push → roster-review issue → `--check-status` → `collectors/vrs.py --check` |
| `live-refresh.yml` | every 15 min | `daily_refresh.py --live` while a tracked event runs → gates → db → push |
| `notes-facts.yml` | daily 06:47 UTC | bundle (for the 3-day scope) → `collectors/notes_fetch.py` (`OPENAI_API_KEY` secret) → bundle → `collectors/notes_summarize.py` (AI "why" summaries) → tests → commit the `notes_*` data files + push; issue on failure |
| `validate.yml` | every 30 min | `checks/validate_freshness.py`; restarts the live refresh when behind |

Both refresh workflows share one concurrency group, so two runs never write
the cache together (`notes-facts.yml` has its own: it writes only the
`data/notes_*` files). Vercel rebuilds on every push using `vercel.json`'s
`buildCommand`; nothing needs installing (stdlib only).

## Running locally

```sh
python -m unittest discover -s tests            # full suite
python viewer/build_bundle.py viewer/data-bundle.json
python viewer/build_viewer.py out/index.html     # open out/index.html
python pipeline/daily_refresh.py --live          # backstop refresh from a home connection
```

---

## Change log

Newest first. One line per change that touches structure, data flow, a data
file, a workflow or a public behaviour of the site.

- **2026-10-15**: Post-match recaps at the top of a finished match's Notes tab: AI-written
  by `collectors/notes_summarize.py` in the house style (cause-first lede, map by map,
  the model check, what's next), rule-based from the same match data otherwise. The
  "why" text is now plain words with no figures and covers every factor.
- **2026-10-15**: Smoother scrolling: the content panels no longer use a backdrop blur
  (re-blurred every scroll frame; 21-29 ms frames -> 16.7 ms); scroll listeners passive.
  The Notes tab and team facts restyled to the site's panels; the notes list is not shown.
- **2026-10-15**: "Why the model picks X" at the top of the Notes tab: AI-written for
  scheduled matches (`collectors/notes_summarize.py`, step in `notes-facts.yml`,
  `data/notes_summaries.json`, $5/month cap), rule-based from the same factor numbers
  otherwise. Teams Liquipedia lists under `{{FormerParticipants}}` (withdrew / replaced)
  are no longer event participants (`viewer/events.py`; Virtus.pro at Thunderpick 2026).
- **2026-10-15**: Matchup notes move into a **Notes** tab (next to Veto simulation on
  the home and match pages, and in the event matchup tool) and gain an always-filled
  fact sheet (`notes.team_sheet` / `h2h_sheet`, tool 1.1.0); team pages show the
  sheet too. The panel under the match court is gone.
- **2026-10-15**: Notes are generated on request: new `api/` folder with the Vercel
  function `api/notes_api.py` (CDN-cached per URL, pinned to the snapshot's
  `inputs_version`); the page ships no notes and fills its panels from it;
  notes panels also on the home matchup and the event matchup tool. Decision
  record and asof semantics: `docs/notes-contracts.md` section 4.
- **2026-10-15**: Matchup notes (display only, never a model input): rule
  engine + tool `model/notes.py` (contracts `docs/notes-contracts.md`, manifest
  `docs/notes-tool-manifest.json`), fact schema `model/notes_facts.py`, tenure
  ported to `model/tenure.py`, bundle section `notes` (`viewer/notes_data.py`,
  page `NOTES`), web-fact fetcher `collectors/notes_fetch.py` + `notes-facts.yml`
  (`data/notes_facts.json`, manifests), `evaluation/notes_eval.py`, leakage test 8.
  Match pages get a "Matchup context" panel, team pages a "Team context" strip.
- **2026-10-14**: Repository reorganised into role folders: root-level model
  code → `model/` and `evaluation/`; code that lived in `data/` → `collectors/`;
  `scripts/` split into `pipeline/` and `checks/`; background originals →
  `assets/backgrounds/`. One `paths.py` replaces the per-file `sys.path` edits.
  Added this document and `tests/test_architecture.py`. Site output unchanged.
- **2026-10-14**: Refresh fetches live pages first; a single Liquipedia refusal
  heals on the next run instead of failing (`--max-stale-minutes 30` on live).
- **2026-10-14**: UX overhaul: dashboard home, match pages, Stats page,
  site background, inactive teams (`data/inactive_teams.json`).
