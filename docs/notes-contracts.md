# Matchup notes: shared contracts (frozen v1)

Three contracts every piece of the notes feature speaks: the fact schema (what
the fetcher writes), the note schema (what the rule engine emits and the page
and tool read) and the tool interface. **A change to any field here bumps that
contract's version and updates this file in the same change.**

Notes are **context, not signal**: no note, fact or notes field is ever read
by `model/predictor.py`, `model/backtest.py` or any weight fit.

---

## 1. Fact schema v1 (`data/notes_facts.json`)

Written only by `collectors/notes_fetch.py`; validated by
`model/notes_facts.py::validate_fact` on write **and** again on read (a fact
that fails is dropped, never repaired).

```json
{
  "fact_schema": 1,
  "updated_at": "2026-10-14T06:00:00Z",
  "form": [ <form fact>, ... ],
  "news": [ <news fact>, ... ]
}
```

`updated_at` is the time of the last fetch run that wrote the file (data, not
a clock read by the engine). Facts are **appended**: a re-fetch adds a new
observation, older observations stay so historical as-of queries can read what
was known then. Exact duplicates (same `id`) are not re-added. Observations
older than 400 days are pruned.

### Common fields (both types)

| Field | Type | Req | Rule |
|---|---|---|---|
| `id` | string | yes | 12-hex sha1 of the fact's identity fields + `observed_at` (`notes_facts.fact_id`) |
| `team` | string | yes | Exact bundle team name (`data.teams` key); the fetcher maps HLTV names before writing |
| `player` | string | yes | Player nickname as shown by the source (case kept) |
| `observed_at` | string | yes | UTC `YYYY-MM-DDTHH:MM:SSZ`: when the fetcher read the source |
| `source_url` | string | yes | `https://` URL of the page the numbers / news came from. No source, no fact |
| `source` | string | yes | Short label shown on the page: `HLTV`, `Liquipedia`, or the outlet's domain |
| `method` | string | yes | `scrape` (HTTP fetch + LLM extractor) or `web_search` (LLM with web search) |

### Player-form fact (`form[]`)

| Field | Type | Req | Rule |
|---|---|---|---|
| `rating_last30` | number | yes | 0.3 ≤ x ≤ 2.5 (HLTV Rating, last 30 days) |
| `rating_baseline` | number | yes | 0.3 ≤ x ≤ 2.5 |
| `baseline` | string | yes | `last12m` or `career` |
| `maps_last30` | int | yes | 0 ≤ n ≤ 80 |
| `rating_version` | string | no | e.g. `2.1`, `3.0` as labelled by the source |

Expiry: an observation counts on cutoff C only when `observed_at < C` and
`C − observed_at ≤ 14 days`. Per (team, player) the newest such observation
is used.

### News fact (`news[]`)

| Field | Type | Req | Rule |
|---|---|---|---|
| `type` | string | yes | `stand_in` or `roster_rumor` |
| `date` | string | yes | `YYYY-MM-DD`: the date the source dates the news / the stand-in starts |
| `replaced` | string | no | `stand_in`: the regular being replaced |
| `until` | string | no | `stand_in`: last day the stand-in applies, `YYYY-MM-DD` |
| `event` | string | no | `stand_in`: event name the stand-in is announced for |
| `move` | string | no | `roster_rumor`: `in` (linked with joining `team`) or `out` (linked with leaving) |
| `other_team` | string | no | `roster_rumor`: the other organisation named |

Expiry on cutoff C (day = C's date): both types need `observed_at < C`.
`stand_in` is active while `date − 3 days ≤ day ≤ until` (no `until`: `date +
21 days`). `roster_rumor` is shown while `day − date < 90 days`. A rumor is
never phrased as fact.

---

## 2. Note schema v1

Every note, whatever surface it appears on:

```json
{
  "id": "5f0c2a91d3b7",
  "rule": "SHORT_REST",
  "scope": "team",
  "team_scope": "Aurora",
  "kind": "fact",
  "headline": "Short rest: last played ESL Pro League Season 24 (Semifinal) 3 days ago",
  "evidence": "Last official: 2026-10-11 · 9 officials in the last 14 days",
  "values": {"days": 3, "event": "ESL Pro League Season 24", "stage": "Semifinal", "last_date": "2026-10-11", "n14": 9},
  "source": "matches",
  "source_url": null,
  "observed_at": "2026-10-11",
  "age_days": 3,
  "rank": 2
}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | string | 12-hex sha1 of `rule|team_scope|values` (stable; duplicates share an id, which is a bug the engine refuses) |
| `rule` | string | One of the rule names in `notes.RULES` |
| `scope` | string | `team` (about one team, any opponent) or `matchup` (about this pairing: head-to-head) |
| `team_scope` | string | The team the note is about (for `matchup` notes, the team the sentence is about) |
| `kind` | string | `fact` or `rumor` (rumors render tagged and dimmed) |
| `headline` | string | One plain line, built from a template in `notes.py`, never by an LLM |
| `evidence` | string | The numbers behind it, `·`-separated |
| `values` | object | The same numbers, machine-readable (what an LLM should quote) |
| `source` | string | `matches` (data/matches.json), `lineups` (data/lineups.json + squads), or a web fact's `source` label |
| `source_url` | string\|null | Web facts only; `null` for bundle-derived notes |
| `observed_at` | string | `YYYY-MM-DD`: date of the newest information the note rests on |
| `age_days` | int | asof day − `observed_at` (≥ 0) |
| `rank` | int | Position of `rule` in the fixed importance order (1 = most important) |

### Importance order (fixed)

1 STAND_IN · 2 SHORT_REST · 3 COLD_REST · 4 TENURE_NEW · 5 NEW_ADDITION ·
6 REVENGE · 7 HOT_PLAYER · 8 COLD_PLAYER · 9 STREAK · 10 HEAVY_SCHEDULE ·
11 TENURE_STABLE · 12 ROSTER_RUMOR

A matchup keeps at most 6 notes: sorted by (`rank`, side order a → b,
magnitude desc, `id`), at most `CONFIG["max_per_rule_team"]` of one rule per
team. Team pages keep at most `CONFIG["max_team_notes"]`.

### Freshness signal (`notes.freshness` in the bundle, and in tool output)

```json
{"as_of": "2026-10-15", "facts_updated_at": "2026-10-14T06:00:00Z",
 "oldest_fact_age_days": 2, "stale": false, "stale_after_days": 7,
 "scope_teams": 18, "teams_with_no_facts": ["9z"], "facts_live": 64}
```

Computed over the **scope teams** (teams with a fixture in the next 7 days).
`oldest_fact_age_days` is the age of the oldest *newest* fact among scope
teams that have any (null when none); `stale` = `oldest_fact_age_days > 7` or
no fact run at all.

---

## 3. Tool interface v1 (`model/notes.py`; manifest `docs/notes-tool-manifest.json`)

```
get_notes(team_a, team_b, asof_date, bundle, match_day=None, event=None) -> dict
get_notes_batch(matchups, asof_date, bundle) -> dict
team_notes(team, asof_date, bundle) -> dict
```

- `bundle`: the parsed `viewer/data-bundle.json` (only `bundle["notes"]` is read).
- `asof_date`: `YYYY-MM-DD` (information cutoff = start of that day UTC: a match
  played that day sees only what was known before it) or
  `YYYY-MM-DDTHH:MM:SSZ` (cutoff = that instant) or `latest` (= the bundle's
  `notes.live_cutoff`: everything known now). **The leakage enforcement
  point**: series dated on or after the cutoff day and facts observed at or after
  the cutoff are invisible.
- `team_a` / `team_b`: bundle name, case-insensitive; short names from
  `data/team_aliases.json` (`nip`, `9z`, `aurora`) resolve to the full name,
  which the output returns.
- `match_day`: day the series is played (rest / schedule counts run to it);
  default = asof day. Never moves the information cutoff.
- `event`: the series' event name; a SHORT_REST at that same event is not shown
  (playing again inside one event is not rest).
- `matchups`: `[{"team_a", "team_b", "match_day"?, "event"?}]`.

Success:

```json
{"tool": "faceoff_get_notes", "tool_version": "1.0.0", "note_schema": 1, "ok": true,
 "team_a": "Aurora", "team_b": "9z", "asof_date": "2026-10-14", "match_day": "2026-10-14",
 "notes": [<note>, ...], "omitted": 2, "freshness": {...},
 "counted_in_win_probability": false}
```

`omitted` = notes that fired but fell outside the 6-note cap. `notes: []` with
`ok: true` means the rules ran and nothing fired, never a failure.

Errors (`ok: false`, never an empty note list):

```json
{"tool": "faceoff_get_notes", "tool_version": "1.0.0", "ok": false,
 "error": {"code": "UNKNOWN_TEAM", "message": "No team named 'Aurorra' in the bundle.",
           "team": "Aurorra", "close_matches": ["Aurora"]}}
```

| `code` | Extra fields |
|---|---|
| `UNKNOWN_TEAM` | `team`, `close_matches` (≤ 5, `difflib`) |
| `SAME_TEAM` | `team` |
| `BAD_DATE` | `field`, `value` |
| `MISSING_FIELDS` | `fields` (dotted bundle paths that are absent) |
| `BAD_SCHEMA` | `found`, `supported` (bundle `notes.schema` versions) |

Batch: `{"tool", "tool_version", "ok": true, "results": [<get_notes output>, ...]}`;
one bad matchup is an error object in its slot, the others still return.

**Versioning**: `tool_version` is semver. A breaking change to the output or
the note schema bumps the major (and `note_schema`); new optional fields bump
the minor.

CLI: `python model/notes.py --a Aurora --b 9z --date 2026-10-14 [--bundle PATH] [--team-only]`.
