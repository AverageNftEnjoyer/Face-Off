# CS2 engine backtest report

Fan analytics only. Real Liquipedia results (data/SOURCES.md). Ratings are a point-in-time **Elo proxy**, not HLTV ratings.

## Data

- series: 2393 ({'1': 361, '3': 1980, '5': 52}) from 2023-10-16 to 2026-10-06, 110 teams
- BO3 series: 1980; excluded (either team < 5 prior series): 324
- evaluated: 1656 -> train 998 (2023-10-22..2025-11-05), test 658 (2025-11-07..2026-10-06)
- |rating gap| quantiles (proxy scale): q25=0.018, q50=0.040, q75=0.075, q90=0.115

## Temperature (fitted on train only)

- T = **0.970** (engine CONFIG['temperature'] fitted on total_logodds (train))
- train log-loss: T=1 0.6289 -> fitted 0.6288

## Test-set metrics

| model | n | accuracy [95% Wilson] | Brier [95% boot] | log-loss [95% boot] | ECE10 | scoreline acc |
|---|---|---|---|---|---|---|
| coin_flip | 658 | 0.500 [0.462, 0.538] | 0.2500 [0.2500, 0.2500] | 0.6931 [0.6931, 0.6931] | 0.0486 | - |
| elo_only | 658 | 0.646 [0.609, 0.681] | 0.2166 [0.2039, 0.2295] | 0.6232 [0.5946, 0.6533] | 0.0269 | - |
| higher_elo_pick | 658 | 0.646 [0.609, 0.681] | n/a | n/a | n/a | - |
| old_engine_v1 | 658 | 0.568 [0.530, 0.606] | 0.3182 [0.2907, 0.3452] | 1.1048 [0.9665, 1.2557] | 0.2677 | 0.260 |
| old_engine_v1_posthoc_T | 658 | 0.568 [0.530, 0.606] | 0.2369 [0.2298, 0.2438] | 0.6649 [0.6489, 0.6803] | 0.0430 | - |
| new_engine | 658 | 0.640 [0.602, 0.676] | 0.2165 [0.2031, 0.2302] | 0.6219 [0.5924, 0.6525] | 0.0420 | 0.375 |
| new_engine_T1 | 658 | 0.640 [0.602, 0.676] | 0.2167 [0.2030, 0.2307] | 0.6224 [0.5918, 0.6536] | 0.0437 | 0.377 |

Notes: coin flip accuracy is credited 0.5 per series by definition. higher_elo_pick is a hard pick (accuracy only). old_engine_v1_posthoc_T is a diagnostic, not the shipped old engine. Scoreline acc = modal BO3 scoreline from series_probs vs actual (test base rates: 2-0 0.32, 2-1 0.23, 1-2 0.19, 0-2 0.26).

## Paired Brier differences (test, 95% paired bootstrap; negative = first model better)

- new_engine - elo_only: -0.0001 [-0.0026, +0.0024]; log-loss diff -0.0013
- new_engine - old_engine_v1: -0.1017 [-0.1250, -0.0791]; log-loss diff -0.4829
- old_engine_v1 - elo_only: +0.1016 [+0.0786, +0.1250]; log-loss diff +0.4816
- new_engine - coin_flip: -0.0335 [-0.0469, -0.0198]; log-loss diff -0.0712

## Reliability table: new_engine (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 0 | - | - |
| 0.1-0.2 | 20 | 0.188 | 0.250 |
| 0.2-0.3 | 68 | 0.247 | 0.265 |
| 0.3-0.4 | 74 | 0.355 | 0.365 |
| 0.4-0.5 | 90 | 0.446 | 0.511 |
| 0.5-0.6 | 115 | 0.549 | 0.470 |
| 0.6-0.7 | 107 | 0.650 | 0.664 |
| 0.7-0.8 | 109 | 0.760 | 0.706 |
| 0.8-0.9 | 75 | 0.811 | 0.840 |
| 0.9-1.0 | 0 | - | - |

## Reliability table: old_engine_v1 (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 103 | 0.032 | 0.427 |
| 0.1-0.2 | 72 | 0.142 | 0.528 |
| 0.2-0.3 | 42 | 0.254 | 0.524 |
| 0.3-0.4 | 34 | 0.352 | 0.382 |
| 0.4-0.5 | 36 | 0.449 | 0.556 |
| 0.5-0.6 | 34 | 0.558 | 0.529 |
| 0.6-0.7 | 38 | 0.647 | 0.500 |
| 0.7-0.8 | 43 | 0.746 | 0.465 |
| 0.8-0.9 | 66 | 0.851 | 0.485 |
| 0.9-1.0 | 190 | 0.967 | 0.711 |

## Reliability table: elo_only (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 1 | 0.063 | 0.000 |
| 0.1-0.2 | 21 | 0.154 | 0.143 |
| 0.2-0.3 | 43 | 0.263 | 0.302 |
| 0.3-0.4 | 73 | 0.353 | 0.342 |
| 0.4-0.5 | 120 | 0.448 | 0.467 |
| 0.5-0.6 | 133 | 0.550 | 0.519 |
| 0.6-0.7 | 120 | 0.646 | 0.658 |
| 0.7-0.8 | 83 | 0.751 | 0.735 |
| 0.8-0.9 | 50 | 0.845 | 0.900 |
| 0.9-1.0 | 14 | 0.926 | 0.714 |

## Reliability tiers / bands: new_engine

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 312 | 0.702 | 0.1923 | 26.5 | 0.711 |
| MEDIUM | 178 | 0.635 | 0.2264 | 31.3 | 0.650 |
| LOW | 168 | 0.530 | 0.2508 | 38.0 | 0.618 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 164 | 21.3-26.7 | 0.1885 | 0.720 |
| Q2 | 165 | 26.7-29.8 | 0.1978 | 0.679 |
| Q3 | 164 | 29.8-33.7 | 0.2285 | 0.634 |
| Q4 | 165 | 33.7-52.6 | 0.2510 | 0.527 |

## Reliability tiers / bands: old_engine_v1

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 0 | - | - | - | - |
| MEDIUM | 45 | 0.578 | 0.2926 | 36.5 | 0.711 |
| LOW | 613 | 0.568 | 0.3200 | 56.8 | 0.845 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 164 | 30.4-49.9 | 0.2578 | 0.616 |
| Q2 | 165 | 50.0-57.3 | 0.2795 | 0.582 |
| Q3 | 164 | 57.4-64.0 | 0.3514 | 0.524 |
| Q4 | 165 | 64.0-64.0 | 0.3838 | 0.552 |

## Overconfidence check (outputs >= 90% or <= 10%)

- old_engine_v1 (test): 293 series (44.5%), hit-rate 0.662, mean stated conf 0.967
- new_engine (test): 0 series (0.0%), hit-rate -, mean stated conf -
- elo_only (test): 15 series (2.3%), hit-rate 0.733, mean stated conf 0.927
- old_engine_v1 (all 1656 eval series): 711 (42.9%), hit-rate 0.640, mean stated conf 0.967

## Feature derivation (point-in-time, only series dated strictly before the match day)

- rating: series Elo (init 1500, K=32 x {1: 0.75, 3: 1.0, 5: 1.25} by best-of), rating = 1 + (elo-1500)/2000. PROXY, not HLTV.
- form30/n30/opp_rating30: series win rate, count and mean opponent proxy rating in the 30 days before; keys omitted if no series.
- form5: win rate over the last <=5 series; omitted if none.
- h2h: series wins vs each other in the prior 365 days.
- maps_a/b: per-map [win rate, maps played] over the prior 90 days.
- map_pool: maps with >= 3 plays (all teams) in the prior 60 days, the 7 most recently played (old engine ignores it: hardcoded pool).
- volatility: |wins - Elo-expected wins| / n over last 10 series x 2.5, clamped [0,1]; 0.3 if < 3 series.
- permaban: None (no veto data); roster: stand-in and missing-IGL flags from Liquipedia team-page stand-in tables (data/rosters.py); stakes: none.

