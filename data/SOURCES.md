# Data sources

Fetched 2026-10-06; tournament discovery by Liquipedia's S/A tier categories
added on 2026-10-07 (see "Tier coverage" below). Only S-tier and A-tier
tournaments are tracked. Every record in `matches.json` comes from a real source page;
nothing is invented, imputed or estimated.

## 1. Liquipedia Counter-Strike: USED (the only source of match records)

- API: https://liquipedia.net/counterstrike/api.php (MediaWiki)
- Terms: https://liquipedia.net/api-terms-of-use. We sent a descriptive
  User-Agent, accepted gzip, waited 2.5 s between `action=query` requests, and
  cached every response. `action=parse` was never used. Team-name resolution
  used 2 `action=expandtemplates` calls, spaced 30 s apart and treated with
  the parse-level limit. In total about 90 requests were made. The tier
  discovery work (2026-10-07) added about 370 `action=query` requests
  (category listings, candidate pages 20 per request, team pages 20 per
  request, imageinfo, prefix listings; roughly half were repeat listings while
  the discovery rules were tuned) and about 270 image downloads. Part of that
  covered B-tier events, which were tracked for one day and then dropped (see
  below); their cached responses stay in `raw/pages/` but nothing reads
  them.
- Content license: CC BY-SA 3.0 (Liquipedia text content). Attribution:
  "Data from Liquipedia (liquipedia.net/counterstrike), CC BY-SA 3.0".
  Derived data in this folder inherits CC BY-SA.
- Raw responses: `raw/pages/*.json` (380 files on 2026-10-07). Each file stores the request URL, the fetch
  time and the full JSON response. The page list is in `raw/lp_titles_selected.txt` (168
  lines). The first 138 were selected from the `allpages` prefix listings in
  `raw/lp_titles_all.txt`; the other 30 come from the tier categories (below).
  Every line is an S- or A-tier page by its own infobox, or a stage page of
  one (a stage page with no tier of its own follows its parent).
  The hand-picked part covers S/A-tier main events: IEM, ESL Pro League S19-S26, BLAST Premier 2023-24,
  BLAST Open/Rivals/Bounty 2025-26, PGL 2024-26, the CS2 Majors (Copenhagen 2024,
  Shanghai 2024, Austin 2025, Budapest 2025, Cologne 2026), Thunderpick WC,
  Esports World Cup and StarSeries. Regional qualifiers and open qualifiers are
  excluded, and so is the C-tier "Esports World Cup/2024/Middle East"
  qualifier, which was on the hand-picked list until 2026-10-07 (6 series).
- Tier coverage (added 2026-10-07). Discovery reads Liquipedia's own tier
  categories, `Category:S-Tier Tournaments` (504 pages), `Category:A-Tier
  Tournaments` (828), with `list=categorymembers` (500 per request, newest
  additions first). A title is a candidate when it names 2026/2027, or names
  no year and was added since 2024-01-01. Each candidate's own infobox
  decides: `liquipediatier` must be S or A (stored per event as `tl`), `liquipediatiertype` must not be Qualifier,
  Showmatch, Weekly, Monthly, Misc or Points, `sdate` must be in 2026/2027 and
  the event at most 45 days long (season-long leagues are left out). Regional
  sub-pages of a tracked event ("Thunderpick/World Championship/2026/North
  America") are regional qualifiers and skipped. A sub-page dated apart from
  its parent with the same tier is its own event (ESL Challenger League cups,
  "BetBoom RUSH B! Summit 2026: Part Deux"); one that overlaps it, names a
  stage type ("Online Stage") or has no tier of its own is a stage of it
  (`events.is_stage`). Result: 30 main events (16 S, 14 A) beyond the
  hand-picked list. `scripts/daily_refresh.py` runs the same discovery every 6
  hours (first 500 entries per tier, 2 requests); pages it finds through the
  tracked-series `allpages` listings are dropped when their own infobox names
  a tier below A.
  B-tier and lower tournaments are not tracked. They were included on
  2026-10-07 (161 main events, 5,551 series) and removed the same day: their
  results are much less predictable, and the site and the engine are about
  S- and A-tier matches. Rating DATA only (not used by the engine or the
  site, no predictions): `python data/collect_liquipedia.py --offline --lower`
  rebuilds `matches_lower.json` from the cached pages listed in
  `raw/lp_titles_lower.txt` (381 pages). Every series has `tier` (the
  page's own infobox) and `kind`:
  `main` (B-tier main events and their stages, 7,240 series), `qualifier`
  (closed regional qualifiers of S/A events, 1,556 series; open
  qualifiers are left out), `sa_gap` (165 series from 14 S/A main events of
  2025 that the hand-picked S/A list does not track, e.g. Roobet Cup 2025,
  CS Asia Championships 2025, MESA Nomadic Masters 2025; kept out of
  `matches.json` so the engine's history is unchanged). Total 8,961 series,
  2025-01 .. 2026-10 (plus the 6-series 2024 EWC Middle East qualifier). A
  series also in `matches.json` (same day, same teams) is left out (none
  overlapped). Offline rebuilds are byte-identical.
- 2025 backfill (2026-10-08, `scripts/backfill_lower.py plan|fetch`):
  discovery from the cached `Category:B-Tier Tournaments` listing (title
  names 2025, or no year and added since 2024-01-01; own infobox: tier B,
  not a qualifier/showmatch/weekly/monthly/misc/points page, sdate in 2025,
  at most 150 days so ESL Challenger League regions count), closed
  qualifiers from the cached `allpages` listings of the tracked S/A events
  (2025 and 2026), S/A gap events from the S/A tier categories. Fetched
  month by month, December back to January, at one request per 5 s:
  20 discovery requests (240 candidate infoboxes, 20 per request), 2
  category-size probes, 54 fetch requests (stage listings and pages), 4
  for the builder (new team codes); 80 in all, no 429 or 5xx answer. Progress
  and the per-month record: `raw/backfill_2025.json`; re-running continues
  from the cache. C-tier (`Category:C-Tier Tournaments`, 15,530 pages,
  ~115 additions a month) was sized and not fetched: about 150-200 requests
  for mostly amateur teams that rarely meet S/A teams.
- Per-event lineups: `python data/event_lineups.py` writes
  `event_lineups.json` from the cached pages (no requests): the
  participant section of each event page ({{TeamParticipants}} /
  {{TeamCard}}), else the team page's squad history dates (join /
  inactive / leave). Coverage: 99.8% of team-series in `matches.json`,
  96.2% in `matches_lower.json`.
- Evidence (roster_rating_check, removed 2026-10-13, in git history, pre-registered,
  `roster_rating_report.json`): on 742 held-out S/A BO3 series after
  2025-10-08, neither lower-tier results as Elo data (K x 0.25) nor a
  roster-aware Elo (regress halfway to 1500 when <= 2 of 5 players carry
  over; graded; with a K boost; combined with the lower-tier data) had a
  log-loss CI below 0, and every variant was worse on one half of the
  window.
- Daily refresh and use (2026-10-13, `scripts/lower_refresh.py`, last step
  of the daily run; docs/pipeline.md "Lower-tier results"): new B-tier events
  from `Category:B-Tier Tournaments` (newest 500 additions, one request),
  closed qualifiers from the cached listings, running pages re-read; budget
  40 requests per run, failures only warn. The first run sent 16 requests
  (1 category listing, 2 candidate infobox batches, 1 stage listing, 2
  qualifier page batches, 8 stage re-listings of running events, 1 page
  refresh, 1 team-code expansion), no 429/5xx, and added 103 series
  (2026-10-07 .. 2026-10-13; 9,064 in all). The site shows the series on team
  pages with their tier label; they never become events or predictions. As
  rating data they wait on `backtest.LOWER_TIER_ELO_WEIGHT`, which is 0: the
  pre-registered test re-run on 2026-10-13 (`lower_tier_report.json`, 581
  primary rows) gave elo_0.25 log loss 0.6140 vs 0.6160 (CI [-0.0092,
  +0.0054]) with calibration passing, but worse than base on the second half
  of the window (0.6255 vs 0.6247), so it stays off.
  Event, team and sponsor names are reproduced
  as Liquipedia writes them.
- Forfeited maps (`score1=W|score2=FF` or the reverse) are now credited to the
  W side; their round scores are from before the forfeit. One older result
  changed with it: HEROIC-NiP 2024-09-03 (ESL
  Pro League S20) goes to NiP, map 1 overturned for illegal equipment.
- What was extracted: each `{{Match}}` template. That gives the opponents
  (team-template codes), the date (event-local, day precision) and each `{{Map}}`:
  the map name, plus the map winner from the half/overtime round scores
  (`t1t+t1ct+o*` vs `t2...`) or from `winner=`/`score1/2`. Maps marked
  `finished=skip` are excluded. The series winner is whichever side reached the
  map-win majority. Best-of is the number of map slots (1/3/5).
- The `finished` flag is typo-tolerant. The cache contains `t`, `trur`, `trie`, `y`
  and `reyw`, and any value other than empty/`skip`/`false` counts. A map still
  needs a winner from `winner=` or round scores. This fix recovered 5 series:
  MOUZ-FURIA 2025-08-01 corrected from 2-0 to 2-1 (Mirage to FURIA), plus 4 series
  that had been dropped: SAW-FaZe 2024-08-16, NaVi-paiN 2025-01-24,
  3DMAX-Astralis 2026-02-15 and MOUZ-IC 2026-08-29.
- Parser and rebuild: `python data/collect_liquipedia.py`. `lpfetch.py` is the polite fetcher.
  Adding `--offline` makes no network calls; a cache miss is an error, so the script
  never writes a partial dataset. Team codes are resolved from every cached
  `expandtemplates` response, independent of request batching. `team_aliases.json`
  is output only and is never read back. Offline rebuilds are byte-identical.

### Counts (`matches.json`)
| | count |
|---|---|
| unique finished series (CS2 era, 2023-10-16 .. 2026-10-06) | 2393 (S-tier 2055, A-tier 338) |
| BO3 / BO1 / BO5 | 1980 / 361 / 52 |
| BO3 without map data | 0 |
| maps with a winner | 5337 |
| teams (after alias resolution) | 110 |
| tournaments on the site (2026 and the next 120 days) | 41 (27 S-tier, 14 A-tier) |
| showmatches dropped | by section (`{{show match}}` / `{{Stage|Showmatch}}`) and by team name |

Every series records the event page it was read from (`event_page`) and, when the
page prints one, its external match number (`ext_match_id`, also used to drop
duplicates).

### Normalization
- Team names: Liquipedia team-template codes (`vit`, `navi`, `tl`, ...) are resolved to the
  team's Liquipedia page name via `{{Team|code}}` expansion (`team_aliases.json`
  lists every code per canonical name, e.g. `FlyQuest <- fq, flyquest`).
  Canonical names are Liquipedia page names ("Team Vitality", "Natus Vincere").
- Maps: "Dust II" becomes "Dust2". Other map names are kept as given.
- Showmatches were dropped: national/all-star/streamer teams (list in `collect_liquipedia.py`,
  `SHOWMATCH_TEAMS`) and any team named "(Showmatch team)".
- Duplicates were removed. Redirected pages, e.g. "IEM 2023/Fall" -> Sydney 2023, appear twice;
  they are de-duplicated on (date, team pair, HLTV id).
- Not available from this source: map veto (bans/picks) and stand-in or roster information.

## 2. Valve `counter-strike_regional_standings`: USED as the SECOND results source (cross-check only)
- https://github.com/ValveSoftware/counter-strike_regional_standings (public repo,
  published by Valve). Also read by `vrs.py` (standings) and `lineups.py`
  (per-series five-man lineups). Raw files are cached in `raw/valve/` (gitignored).
- Access and terms relied on: the repository is public and Valve's readme invites
  the community to use its code and data ("feel free to tinker and make something
  new"). It is read through GitHub's REST API (one `git/trees` call per run,
  authenticated with the workflow's `GITHUB_TOKEN` in Actions, inside GitHub's API
  rate limits) and `raw.githubusercontent.com` downloads spaced 0.25 s, with a
  descriptive User-Agent, every file cached and never re-downloaded. That is
  ordinary use under the GitHub Terms of Service. The repository has no LICENSE
  file, so its files are NOT republished: the committed `valve_results.json`
  holds only derived facts (date, the two teams, the winner, which snapshot listed
  it). Each detail page notes that Valve's event data is "provided by HLTV.org":
  the second source is independent of Liquipedia, but its upstream is HLTV's
  event data as published by Valve. HLTV itself is never contacted.
- What it contains: VRS standings snapshots (`live/<year>/`, monthly since
  2025-03, more often before; 47 detail snapshots in the tree on 2026-10-12) and,
  per snapshot, one detail page per ranked roster listing the series that fed it
  over the previous six months: a snapshot-local match id (the same id on both
  rosters' pages), date (day precision), opponent display name, W/L and the five
  players. No maps, no best-of, no score, no event name.
- Second results source (added 2026-10-12): `valve_results.py` joins the rows into
  one record per series (both pages must agree on the winner, else the series is
  dropped and counted) in the `matches.json` record shape with every field the
  source lacks set to null, and writes `valve_results.json` (6,826 series between
  tracked teams, 2024-02-08 .. 2026-10-04, from the 14 cached snapshots).
  Team names are derived, never listed by hand: our names from `matches.json` /
  `matches_lower.json` plus every code in `team_aliases.json`, compared after
  `lineups.norm`; a normalised key claimed by two teams is dropped as ambiguous.
  Names that still do not resolve are inferred from dates only (never from who
  won), with the thresholds in the tolerance block of `valve_results.py`; this
  found two: `Betclic -> Betclic Apogee Esports`, `Imperial Valkyries -> Imperial
  Female`. The refresh updates it incrementally (only a newer snapshot is read; a
  failure only warns). This work made no new requests: everything was built from
  the cache `lineups.py` had already downloaded.
- Measured against `matches.json` (`scripts/cross_source_check.py`,
  `cross_source_report.json`, overlap 2024-02-08 .. 2026-10-04):
  2,227 primary series in the overlap; 2,191 matched (98.4%) and **all 2,191
  agree on the winner (0 disagreements)**; 16 pending (ESL Pro League Season 24
  was still running when the newest snapshot was cut); 17 identity mismatches (same day, same team, same
  result, different opponent): 9 series that `matches.json` gives to "Monte" at
  IEM Cologne Major 2026 are "Luminosity" in Valve's data, and 8 that it gives to
  "ECSTATIC" at PGL Major Copenhagen 2024 are "Gaimin Gladiators". These need a
  human check against the event pages; nothing was changed automatically.
  3 primary series are missing from the second source (HEROIC-NiP at BLAST
  Spring Showdown 2024-03-06, and two Asian Champions League 2026 series of
  "Last Bullet", whose Valve name could not be placed). The second source also
  has 4,635 series between tracked teams that `matches.json` lacks: 1,607 are in
  `matches_lower.json` (B-tier, deliberately untracked), 3,028 are other
  untracked events.
- Freshness: it cannot keep results current. Valve publishes a snapshot about
  once a month (first Monday; median gap 12 days over the whole tree, 28-35 days
  since March 2025) and apparently lists a series only once its event has
  finished (ESL Pro League Season 24, running on 2026-10-05, is absent from that
  snapshot while finished events up to 2026-10-04 are in it). Measured
  from the cached snapshots, a series first appeared a median 36 days (p90 69)
  after it was played (an upper bound: the initial cache holds every other
  snapshot; incremental runs read every new one). With the newest snapshot
  dated 2026-10-05, the latest series it has is 2026-10-04, a week behind
  `matches.json` (2026-10-11). It is a cross-check, not a backup feed.

## 3. Not used
- HLTV: not scraped (no official API; bot protection; the project does not scrape
  it). Its data reaches us only as Valve republishes it in section 2.
- Commercial esports data APIs (PandaScore, GRID, Abios): they need an account /
  API key and contract terms the project does not have; not evaluated further.
- Steam Web API: no professional match results. FACEIT Data API: only
  FACEIT-hosted matches, not the tracked S/A-tier events.
- Kaggle / Hugging Face CS2 datasets: not needed once Liquipedia worked. None was
  downloaded, so no third-party license applies.
