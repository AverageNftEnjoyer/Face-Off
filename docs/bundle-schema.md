# data-bundle.json schema (v1)

`viewer/data-bundle.json` is the single versioned artifact every frontend surface
renders from. It is produced by `python viewer/build_bundle.py` (which runs
`viewer/build_viewer.py::build_data()`), and it is a pure function of
`data/matches.json` + cached Liquipedia pages + `viewer/assets.json`.
Nothing in it is estimated by hand.

The bundle is **gitignored** — it is rebuilt by CI (both refresh workflows)
and by Vercel's `buildCommand` on every deploy, so the deployed page can
never serve a stale bundle. `viewer/build_viewer.py` renders exclusively
from the bundle and fails loud if it is missing, corrupt, or a foreign
version.

**Contract rule:** the backend may only change this schema by bumping
`bundle_version` AND updating this document in the same commit. The frontend
must refuse to render a bundle whose `bundle_version` it does not understand.

## Envelope

| Field | Type | Meaning |
|---|---|---|
| `bundle_version` | int | Schema version. `1` for this document. |
| `generated_at` | string | UTC build timestamp, `YYYY-MM-DDTHH:MM:SSZ`. Pinned via `FACEOFF_GENERATED_AT` for byte-identical rebuilds. |
| `data_through` | string | Last match date ingested, `YYYY-MM-DD`. The honest staleness label — the site header renders "data through {data_through}". |
| `data_hash` | string | 12-hex sha1 of `data/matches.json` (same stamp the freshness check reads off the deployed page). |
| `vrs_date` | string\|null | Valve Regional Standings date used, `YYYY-MM-DD`. |
| `fixtures` | array | Upcoming matches with predictions (below). |
| `results` | array | Recently finished matches with scores + verdicts (below). |
| `data` | object | Full computed payload (below). |

## fixtures[]

Upcoming matches: both teams known, not finished, on a tracked event page.
Sorted by `(day, when, event_slug)`. Each:

```json
{
  "fixture_id": "9f3c1a2b4d5e",
  "event": "ESL Pro League Season 24",
  "event_slug": "esl-pro-league-season-24",
  "stage": "Semifinal",
  "day": "2026-10-10",
  "when": "13:25",
  "best_of": 3,
  "team_a": "Team Spirit",
  "team_b": "MOUZ",
  "prediction": { ... }
}
```

`fixture_id` is deterministic: sha1 of `event_slug|day|team_a|team_b`, 12 hex.
`team_a` is the reference side for every probability in `prediction`.

### prediction

```json
{
  "win_prob_a": 0.6769,
  "ci_90": [0.507, 0.810],
  "scoreline_probs": {"2-0": 0.3433, "2-1": 0.3336, "1-2": 0.1923, "0-2": 0.1308},
  "modal_scoreline": "2-0",
  "veto": [
    {"team": "A", "action": "ban", "map": "Train"},
    {"team": "B", "action": "ban", "map": "Inferno"},
    {"team": "A", "action": "pick", "map": "Nuke"},
    {"team": "B", "action": "pick", "map": "Dust2"},
    {"team": "A", "action": "ban", "map": "Cache"},
    {"team": "B", "action": "ban", "map": "Mirage"},
    {"team": null, "action": "decider", "map": "Ancient"}
  ],
  "veto_maps": ["Dust2", "Mirage", "Inferno", "Nuke", "Ancient", "Anubis", "Train"],
  "map_win_prob_a": [0.77, 0.52, 0.61, 0.47, 0.62, 0.55, 0.58],
  "factors": {
    "base_strength": 19.4, "form_30d": -1.2, "form_last5": -0.3,
    "head_to_head": 0.1, "map_veto": 0.0, "roster": 0.0, "map_depth": 0.0
  },
  "warnings": [],
  "reliability": "HIGH",
  "pre_match_elo": [1808, 1730]
}
```

- `win_prob_a`: P(team_a wins the series). The point estimate — the headline.
- `ci_90`: 90% confidence interval for team_a's win probability. **This is the
  real prediction; the point estimate is the headline.** Wide intervals mean
  hedged language; the frontend must never present the point without the band.
- `scoreline_probs`: from team_a's perspective. Keys are fixed: `2-0`, `2-1`,
  `1-2`, `0-2`. (BO3 output fields are frozen; a BO5 extension would add a
  separate six-scoreline structure, never reuse these keys.)
- `modal_scoreline`: argmax of `scoreline_probs`.
- `veto`: the simulated veto in order. `team` is `"A"`/`"B"` (team_a/team_b) or
  `null` for the decider. `action` is `ban`, `pick`, or `decider`.
- `veto_maps` / `map_win_prob_a`: the map pool in order and team_a's per-map win
  probability on each.
- `factors`: marginal percentage-point contributions, in fixed key order. Positive
  favors team_a.
- `warnings`: engine warning strings (e.g. `WIDE BAND (...)`, `HIGH VOLATILITY:
  ...`). Empty means a clean read.
- `pre_match_elo`: point-in-time Elo of [team_a, team_b] before the match.

## results[]

Finished matches from the last 30 days (cap 150), newest first. Each:

```json
{
  "event": "ESL Pro League Season 24",
  "event_slug": "esl-pro-league-season-24",
  "stage": "Quarterfinal",
  "day": "2026-10-09",
  "team_a": "Team Falcons",
  "team_b": "Team Spirit",
  "series_score": "0-2",
  "winner": "Team Spirit",
  "maps": [
    {"map": "Mirage", "score_a": 8, "score_b": 13, "winner": "Team Spirit"},
    {"map": "Anubis", "score_a": 10, "score_b": 13, "winner": "Team Spirit"}
  ],
  "verdict": {"predicted": "Team Spirit", "win_prob": 0.429, "hit": true}
}
```

- `series_score`: maps won by team_a-team_b (`w1-w2`).
- `score_a`/`score_b`: round scores from team_a's perspective; `null` when the
  page gave a winner without round scores (e.g. forfeit).
- `verdict`: the model's **pre-match** call for this fixture (point-in-time,
  computed before the match day). `predicted` is the pick, `win_prob` is
  P(team_a) as stored, `hit` is whether the pick won. `null` when no pre-match
  call existed (e.g. a team with no history).

## data (full payload)

The complete computed state. Key sections:

| Key | Contents |
|---|---|
| `as_of` | Ratings date (`data_through + 1 day`), ISO date. |
| `events` | Every 2026 tracked tournament: info, participants, schedule + results with per-map scores, group tables, playoff brackets, pre-match `pred` on each match (compact form). |
| `teams` | Per team: `slug`, `elo`, `rank`, `color`, `vrs` (rank/points/region), `series`, `roster`, `form30`, `maps` (90d map win rates), `results` (last 10 with scores), `trace` (Elo history), `events`. |
| `players` | Player id → name/flag/IGL. |
| `pairs` / `epairs` | All-pairs matchup predictions (compact engine form) for the comparer grids. |
| `h2h` | `"Team A\|Team B"` → last 6 series with scores. |
| `recent` | Last 20 held-out-style pre-match calls with outcomes (track-record feed). |
| `report` | Backtest summary: accuracy/Brier/calibration per model. |
| `pool`, `vmaps`, `vcfg`, `factors`, `map_info` | Map pool, veto coding tables, factor order. |

The compact per-match `pred` inside `data.events[].matches[]` uses short keys;
`fixtures[].prediction` is its expanded, documented form. Both derive from the
same engine call — they cannot disagree.

## Staleness contract

- `data_through` is always present and honest: it is the max series date in
  `data/matches.json` at build time.
- The frontend renders it verbatim ("data through 2026-10-09"). It must never be
  hidden, rounded, or replaced with "live".
- If a build fails, the previous bundle stays live with its age showing.
  Empty/broken is the only unforgivable state.
