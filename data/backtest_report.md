# CS2 engine backtest report

Fan analytics only. Real Liquipedia results (data/SOURCES.md). Ratings are a point-in-time **Elo proxy**, not HLTV ratings.

## Data

- series: 2161 ({'1': 303, '3': 1814, '5': 44}) from 2023-10-16 to 2026-10-05, 100 teams
- BO3 series: 1814; excluded (either team < 5 prior series): 286
- evaluated: 1528 -> train 917 (2023-10-22..2025-10-08), test 611 (2025-10-10..2026-10-05)
- |rating gap| quantiles (proxy scale): q25=0.018, q50=0.041, q75=0.077, q90=0.119

## Temperature (fitted on train only)

- T = **0.909** (engine CONFIG['temperature'] fitted on total_logodds (train))
- train log-loss: T=1 0.6332 -> fitted 0.6327

## Test-set metrics

| model | n | accuracy [95% Wilson] | Brier [95% boot] | log-loss [95% boot] | ECE10 | scoreline acc |
|---|---|---|---|---|---|---|
| coin_flip | 611 | 0.500 [0.461, 0.540] | 0.2500 [0.2500, 0.2500] | 0.6931 [0.6931, 0.6931] | 0.0663 | - |
| elo_only | 611 | 0.661 [0.623, 0.698] | 0.2114 [0.1977, 0.2252] | 0.6122 [0.5802, 0.6448] | 0.0299 | - |
| higher_elo_pick | 611 | 0.661 [0.623, 0.698] | n/a | n/a | n/a | - |
| old_engine_v1 | 611 | 0.599 [0.560, 0.637] | 0.2956 [0.2673, 0.3232] | 1.0126 [0.8987, 1.1172] | 0.2480 | 0.280 |
| old_engine_v1_posthoc_T | 611 | 0.599 [0.560, 0.637] | 0.2287 [0.2205, 0.2366] | 0.6456 [0.6280, 0.6625] | 0.0431 | - |
| new_engine | 611 | 0.661 [0.623, 0.698] | 0.2141 [0.2005, 0.2278] | 0.6177 [0.5874, 0.6482] | 0.0271 | 0.401 |
| new_engine_T1 | 611 | 0.661 [0.623, 0.698] | 0.2144 [0.1998, 0.2292] | 0.6188 [0.5856, 0.6521] | 0.0372 | 0.403 |

Notes: coin flip accuracy is credited 0.5 per series by definition. higher_elo_pick is a hard pick (accuracy only). old_engine_v1_posthoc_T is a diagnostic, not the shipped old engine. Scoreline acc = modal BO3 scoreline from series_probs vs actual (test base rates: 2-0 0.33, 2-1 0.23, 1-2 0.19, 0-2 0.24).

## Paired Brier differences (test, 95% paired bootstrap; negative = first model better)

- new_engine - elo_only: +0.0026 [-0.0017, +0.0072]; log-loss diff +0.0054
- new_engine - old_engine_v1: -0.0816 [-0.1010, -0.0622]; log-loss diff -0.3949
- old_engine_v1 - elo_only: +0.0842 [+0.0624, +0.1059]; log-loss diff +0.4004
- new_engine - coin_flip: -0.0359 [-0.0495, -0.0222]; log-loss diff -0.0755

## Reliability table: new_engine (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 0 | - | - |
| 0.1-0.2 | 14 | 0.157 | 0.357 |
| 0.2-0.3 | 48 | 0.256 | 0.271 |
| 0.3-0.4 | 86 | 0.352 | 0.291 |
| 0.4-0.5 | 82 | 0.453 | 0.524 |
| 0.5-0.6 | 107 | 0.550 | 0.551 |
| 0.6-0.7 | 96 | 0.649 | 0.646 |
| 0.7-0.8 | 125 | 0.753 | 0.752 |
| 0.8-0.9 | 53 | 0.825 | 0.849 |
| 0.9-1.0 | 0 | - | - |

## Reliability table: old_engine_v1 (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 110 | 0.032 | 0.400 |
| 0.1-0.2 | 49 | 0.149 | 0.347 |
| 0.2-0.3 | 36 | 0.248 | 0.583 |
| 0.3-0.4 | 31 | 0.352 | 0.613 |
| 0.4-0.5 | 40 | 0.451 | 0.550 |
| 0.5-0.6 | 29 | 0.545 | 0.621 |
| 0.6-0.7 | 34 | 0.656 | 0.529 |
| 0.7-0.8 | 45 | 0.747 | 0.489 |
| 0.8-0.9 | 56 | 0.852 | 0.589 |
| 0.9-1.0 | 181 | 0.975 | 0.729 |

## Reliability table: elo_only (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 3 | 0.087 | 0.000 |
| 0.1-0.2 | 23 | 0.157 | 0.174 |
| 0.2-0.3 | 33 | 0.253 | 0.303 |
| 0.3-0.4 | 65 | 0.350 | 0.323 |
| 0.4-0.5 | 104 | 0.447 | 0.481 |
| 0.5-0.6 | 123 | 0.549 | 0.528 |
| 0.6-0.7 | 104 | 0.649 | 0.692 |
| 0.7-0.8 | 87 | 0.750 | 0.736 |
| 0.8-0.9 | 54 | 0.842 | 0.870 |
| 0.9-1.0 | 15 | 0.928 | 0.867 |

## Reliability tiers / bands: new_engine

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 279 | 0.706 | 0.1997 | 30.6 | 0.721 |
| MEDIUM | 135 | 0.667 | 0.2178 | 36.0 | 0.638 |
| LOW | 197 | 0.594 | 0.2318 | 41.0 | 0.609 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 152 | 17.6-31.7 | 0.1983 | 0.704 |
| Q2 | 153 | 31.8-35.0 | 0.2035 | 0.699 |
| Q3 | 153 | 35.0-38.5 | 0.2103 | 0.673 |
| Q4 | 153 | 38.5-53.1 | 0.2440 | 0.569 |

## Reliability tiers / bands: old_engine_v1

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 0 | - | - | - | - |
| MEDIUM | 56 | 0.607 | 0.2707 | 35.6 | 0.718 |
| LOW | 555 | 0.598 | 0.2981 | 56.3 | 0.852 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 152 | 26.6-47.3 | 0.2703 | 0.586 |
| Q2 | 153 | 47.4-56.0 | 0.2612 | 0.647 |
| Q3 | 153 | 56.1-64.0 | 0.3175 | 0.582 |
| Q4 | 153 | 64.0-64.0 | 0.3334 | 0.582 |

## Overconfidence check (outputs >= 90% or <= 10%)

- old_engine_v1 (test): 291 series (47.6%), hit-rate 0.680, mean stated conf 0.972
- new_engine (test): 0 series (0.0%), hit-rate -, mean stated conf -
- elo_only (test): 18 series (2.9%), hit-rate 0.889, mean stated conf 0.925
- old_engine_v1 (all 1528 eval series): 708 (46.3%), hit-rate 0.664, mean stated conf 0.971

## Feature derivation (point-in-time, only series dated strictly before the match day)

- rating: series Elo (init 1500, K=32 x {1: 0.75, 3: 1.0, 5: 1.25} by best-of), rating = 1 + (elo-1500)/2000. PROXY, not HLTV.
- form30/n30/opp_rating30: series win rate, count and mean opponent proxy rating in the 30 days before; keys omitted if no series.
- form5: win rate over the last <=5 series; omitted if none.
- h2h: series wins vs each other in the prior 365 days.
- maps_a/b: per-map [win rate, maps played] over the prior 90 days.
- map_pool: maps with >= 3 plays (all teams) in the prior 60 days, the 7 most recently played (old engine ignores it: hardcoded pool).
- volatility: |wins - Elo-expected wins| / n over last 10 series x 2.5, clamped [0,1]; 0.3 if < 3 series.
- permaban: None (no veto data); roster: {} (no stand-in data); stakes: none.

