# Roster core (veto Phase 4) and manual roster overrides: PAUSED

Status: **paused on 2026-10-09 by Jack** to focus on the Simulate Map Picks tab.
Nothing here is switched on. `veto_use_roster` is `False` in `CONFIG` and the live
engine is unaffected.

## 1. What this is, in plain English

A team's map habits belong to its players, not its name. When a team rebrands, signs a
new core, or is new to the data, its own 90-day map record is tiny (PARIVISION: 13
series, several maps with 3 plays). The **roster core** is the team's regular five
players (the five with the most appearances in its last 8 lineups, from
`lineup_features.py`). Phase 4 lets a team with a thin record borrow map evidence from
earlier series where at least 3 of those five played together under any team name.

Shrinkage order (design ADR-5): **team record, then roster core, then the whole field**.
More evidence always makes the estimate sharper and less evidence flatter (invariant I14).

It would mostly help: rebrands, new rosters, stand-ins. It is the phase least likely to
move the headline numbers, because few teams have history thin enough for it to matter.

## 2. Where the design lives

- Architecture: `scratchpad/veto-v2/veto_architecture.html` (published as the private
  artifact "Faceoff Veto Architecture v2"). Sections G (thin samples), ADR-5, Phase P4.
- Reports: `data/veto_phase*_report.json`, `data/veto_fit_report.json`.

## 3. What already exists in the code (scaffolding, inert by default)

Added by the engine agent; verify before relying on any of it.

- `backtest.py`: `HabitAccumulator(core_fn=...)`, `recs_all` (date, five, counts) for the
  roster-core level; `History.core_five(team, dt, core_n=8)`.
- `veto.py` / `predictor.py`: `CONFIG["veto_use_roster"] = False`, `veto_roster_k`.
- `scripts/veto_fit.py`: a roster variant of the fit (`core_fn=h.core_five`).
- `leakage_check.py`: a roster variant that turns `veto_use_roster` on.

No Phase 4 report had been written when work stopped.

## 4. What is still required to finish Phase 4

1. Run the roster fit (grid over `veto_roster_k`) on the training window only
   (dates up to 2025-11-05). Never tune on the test window (>= 2025-11-07).
2. Run the gates against **both** references: the comfort sim and the same-model
   ablation (only temperature and pi fitted). A phase passes only if:
   - pick-set hit improves with a 95% CI lower bound above 0;
   - veto log-likelihood improves with a 95% CI lower bound above 0;
   - scoreline log loss: CI upper bound of (new minus reference) below +0.002;
   - 80%-set coverage in [0.75, 0.85] and map ECE at most 0.03;
   - zero probability-zero rows (`zero_prob == 0`);
   - `p_a` bit-identical on every held-out row.
3. Run `leakage_check.py` with the roster variant. The lineup-core evidence must be
   bit-identical under SCRAMBLE and TRUNCATE, and a lax canary must fail.
4. Skip the roster level for any team whose last lineup is older than its last series
   (warning `lineup_stale_X`).
5. Full test suite (`python -m unittest discover -s tests`, about 70 s).
6. Ship only if every gate passes. Otherwise leave `veto_use_roster = False`.

## 5. Blockers and risks

- **Lineups are stale.** `data/lineups.json` now covers 2,145 series through 2026-10-04.
  Series from Oct 5 on fill in when Valve publishes its next snapshot (about early
  November). A few events (XSE, Logitech G Play, Asian Champions League, Stake Ranked)
  never appear on Valve's pages.
- **Earlier evidence is not encouraging.** The earlier roster-reset test for win
  probability failed, so Phase 4 gets no benefit of the doubt.
- **Small effect expected.** Few teams are thin enough to matter, and the test window
  has already been looked at five times (see `test_evaluation_log` in the fit report).

## 6. Manual roster overrides (planned, not built)

Jack's rule: roster moves are applied **by hand when Jack says so**, not by a workflow.
The Valve snapshot and Liquipedia scrapes rarely catch moves in time.

Planned file: `data/roster_overrides.json`, one entry per move:

```json
{ "team": "PARIVISION", "from": "2026-09-15", "players": ["HObbit", "FL1T", "..."], "kind": "roster_change", "note": "new core" }
```

Rules:

- Entries apply **from their date forward only** (point-in-time), so the backtest and
  `leakage_check.py` stay valid. Never back-fill an override onto earlier matches.
- An override wins over Valve's lineup for that team on those dates.
- `kind` is `roster_change` (permanent) or `stand_in` (temporary); they behave
  differently in the stand-in and continuity features.
- No team or player names in engine code. Everything lives in the file.
- The check script must list unknown or misspelled player names instead of ignoring them.
- Feeds: the roster-core level, the stand-in flag and the continuity feature in
  `lineup_features.py`. The site's roster display (from Liquipedia team pages through
  `viewer/fetch_assets.py`) is a separate source; check whether it needs the same file.
- Needed before building: a leakage test for overrides (an override dated at or after D
  must not change any series dated D) and a unit test for the date cut-off.

How to use it once built: tell Claude "team, date, players". Claude edits the file.
Memory note: `roster-moves-manual.md`.

## 7. Resume checklist

- [ ] Read this file and the architecture document (sections G, ADR-5, P4).
- [ ] Check which of the scaffolding in section 3 still exists and still passes tests.
- [ ] Re-run the lineup refresh (`python data/lineups.py --incremental`) and note coverage.
- [ ] Build `roster_overrides.json` plus its loader, leakage test and unit tests.
- [ ] Run the Phase 4 fit, gates and leakage check (section 4).
- [ ] Decide ship or keep off; record the result in `data/veto_phase4_report.json`.

Rules that stay in force: never commit or push (Jack pushes), stdlib only, no gambling
language, no hardcoded team or map names, frozen `predict_match` fields (add only).
