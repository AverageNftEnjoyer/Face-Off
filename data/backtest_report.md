# CS2 engine backtest report

Fan analytics only. Real Liquipedia results (data/SOURCES.md). Ratings are a point-in-time **Elo proxy**, not HLTV ratings.

## Data

- series: 2167 ({'1': 303, '3': 1820, '5': 44}) from 2023-10-16 to 2026-10-06, 100 teams
- BO3 series: 1820; excluded (either team < 5 prior series): 287
- evaluated: 1533 -> train 921 (2023-10-22..2025-10-10), test 612 (2025-10-11..2026-10-06)
- |rating gap| quantiles (proxy scale): q25=0.018, q50=0.042, q75=0.077, q90=0.119

## Temperature (fitted on train only)

- T = **0.989** (engine CONFIG['temperature'] fitted on total_logodds (train))
- train log-loss: T=1 0.6269 -> fitted 0.6269

## Test-set metrics

| model | n | accuracy [95% Wilson] | Brier [95% boot] | log-loss [95% boot] | ECE10 | scoreline acc |
|---|---|---|---|---|---|---|
| coin_flip | 612 | 0.500 [0.461, 0.539] | 0.2500 [0.2500, 0.2500] | 0.6931 [0.6931, 0.6931] | 0.0670 | - |
| elo_only | 612 | 0.657 [0.618, 0.693] | 0.2121 [0.1995, 0.2253] | 0.6136 [0.5840, 0.6439] | 0.0340 | - |
| higher_elo_pick | 612 | 0.657 [0.618, 0.693] | n/a | n/a | n/a | - |
| old_engine_v1 | 612 | 0.574 [0.534, 0.612] | 0.3040 [0.2781, 0.3321] | 1.0074 [0.9102, 1.1153] | 0.2514 | 0.261 |
| old_engine_v1_posthoc_T | 612 | 0.574 [0.534, 0.612] | 0.2340 [0.2272, 0.2412] | 0.6581 [0.6437, 0.6734] | 0.0508 | - |
| new_engine | 612 | 0.657 [0.618, 0.693] | 0.2118 [0.1986, 0.2259] | 0.6120 [0.5826, 0.6438] | 0.0317 | 0.394 |
| new_engine_T1 | 612 | 0.657 [0.618, 0.693] | 0.2118 [0.1985, 0.2261] | 0.6121 [0.5824, 0.6442] | 0.0306 | 0.394 |

Notes: coin flip accuracy is credited 0.5 per series by definition. higher_elo_pick is a hard pick (accuracy only). old_engine_v1_posthoc_T is a diagnostic, not the shipped old engine. Scoreline acc = modal BO3 scoreline from series_probs vs actual (test base rates: 2-0 0.33, 2-1 0.23, 1-2 0.20, 0-2 0.24).

## Paired Brier differences (test, 95% paired bootstrap; negative = first model better)

- new_engine - elo_only: -0.0004 [-0.0031, +0.0022]; log-loss diff -0.0015
- new_engine - old_engine_v1: -0.0922 [-0.1159, -0.0700]; log-loss diff -0.3954
- old_engine_v1 - elo_only: +0.0918 [+0.0696, +0.1151]; log-loss diff +0.3938
- new_engine - coin_flip: -0.0382 [-0.0514, -0.0241]; log-loss diff -0.0811

## Reliability table: new_engine (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 0 | - | - |
| 0.1-0.2 | 30 | 0.187 | 0.233 |
| 0.2-0.3 | 54 | 0.253 | 0.278 |
| 0.3-0.4 | 66 | 0.359 | 0.379 |
| 0.4-0.5 | 75 | 0.449 | 0.507 |
| 0.5-0.6 | 101 | 0.549 | 0.505 |
| 0.6-0.7 | 91 | 0.651 | 0.637 |
| 0.7-0.8 | 105 | 0.757 | 0.733 |
| 0.8-0.9 | 90 | 0.812 | 0.844 |
| 0.9-1.0 | 0 | - | - |

## Reliability table: old_engine_v1 (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 97 | 0.032 | 0.433 |
| 0.1-0.2 | 48 | 0.144 | 0.417 |
| 0.2-0.3 | 43 | 0.253 | 0.558 |
| 0.3-0.4 | 40 | 0.347 | 0.525 |
| 0.4-0.5 | 44 | 0.453 | 0.614 |
| 0.5-0.6 | 27 | 0.556 | 0.519 |
| 0.6-0.7 | 40 | 0.645 | 0.525 |
| 0.7-0.8 | 47 | 0.753 | 0.574 |
| 0.8-0.9 | 68 | 0.854 | 0.515 |
| 0.9-1.0 | 158 | 0.970 | 0.734 |

## Reliability table: elo_only (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 3 | 0.087 | 0.000 |
| 0.1-0.2 | 23 | 0.157 | 0.174 |
| 0.2-0.3 | 32 | 0.255 | 0.312 |
| 0.3-0.4 | 64 | 0.350 | 0.328 |
| 0.4-0.5 | 105 | 0.447 | 0.486 |
| 0.5-0.6 | 123 | 0.549 | 0.520 |
| 0.6-0.7 | 106 | 0.650 | 0.689 |
| 0.7-0.8 | 88 | 0.749 | 0.727 |
| 0.8-0.9 | 53 | 0.843 | 0.887 |
| 0.9-1.0 | 15 | 0.928 | 0.867 |

## Reliability tiers / bands: new_engine

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 283 | 0.728 | 0.1885 | 26.6 | 0.718 |
| MEDIUM | 155 | 0.645 | 0.2189 | 31.3 | 0.657 |
| LOW | 174 | 0.552 | 0.2434 | 38.1 | 0.635 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 153 | 21.4-27.0 | 0.1870 | 0.719 |
| Q2 | 153 | 27.0-29.9 | 0.2005 | 0.699 |
| Q3 | 153 | 29.9-34.3 | 0.2117 | 0.667 |
| Q4 | 153 | 34.3-53.3 | 0.2479 | 0.542 |

## Reliability tiers / bands: old_engine_v1

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 0 | - | - | - | - |
| MEDIUM | 54 | 0.481 | 0.2968 | 35.6 | 0.702 |
| LOW | 558 | 0.582 | 0.3047 | 56.5 | 0.837 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 153 | 27.7-47.8 | 0.2743 | 0.556 |
| Q2 | 153 | 47.9-56.4 | 0.2788 | 0.569 |
| Q3 | 153 | 56.5-64.0 | 0.3071 | 0.608 |
| Q4 | 153 | 64.0-64.0 | 0.3557 | 0.562 |

## Overconfidence check (outputs >= 90% or <= 10%)

- old_engine_v1 (test): 255 series (41.7%), hit-rate 0.671, mean stated conf 0.969
- new_engine (test): 0 series (0.0%), hit-rate -, mean stated conf -
- elo_only (test): 18 series (2.9%), hit-rate 0.889, mean stated conf 0.925
- old_engine_v1 (all 1533 eval series): 640 (41.7%), hit-rate 0.650, mean stated conf 0.968

## Feature derivation (point-in-time, only series dated strictly before the match day)

- rating: series Elo (init 1500, K=32 x {1: 0.75, 3: 1.0, 5: 1.25} by best-of), rating = 1 + (elo-1500)/2000. PROXY, not HLTV.
- form30/n30/opp_rating30: series win rate, count and mean opponent proxy rating in the 30 days before; keys omitted if no series.
- form5: win rate over the last <=5 series; omitted if none.
- h2h: series wins vs each other in the prior 365 days.
- maps_a/b: per-map [win rate, maps played] over the prior 90 days.
- map_pool: maps with >= 3 plays (all teams) in the prior 60 days, the 7 most recently played (old engine ignores it: hardcoded pool).
- volatility: |wins - Elo-expected wins| / n over last 10 series x 2.5, clamped [0,1]; 0.3 if < 3 series.
- permaban: None (no veto data); roster: stand-in and missing-IGL flags from Liquipedia team-page stand-in tables (data/rosters.py); stakes: none.

