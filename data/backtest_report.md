# CS2 engine backtest report

Fan analytics only. Real Liquipedia results (data/SOURCES.md). Ratings are a point-in-time **Elo proxy**, not HLTV ratings.

## Data

- series: 2396 ({'1': 361, '3': 1983, '5': 52}) from 2023-10-16 to 2026-10-07, 110 teams
- BO3 series: 1983; excluded (either team < 5 prior series): 324
- evaluated: 1659 -> train 998 (2023-10-22..2025-11-05), test 661 (2025-11-07..2026-10-07)
- |rating gap| quantiles (proxy scale): q25=0.021, q50=0.046, q75=0.081, q90=0.124

## Temperature (fitted on train only)

- T = **0.877** (engine CONFIG['temperature'] fitted on total_logodds (train))
- train log-loss: T=1 0.6329 -> fitted 0.6318

## Test-set metrics

| model | n | accuracy [95% Wilson] | Brier [95% boot] | log-loss [95% boot] | ECE10 | scoreline acc |
|---|---|---|---|---|---|---|
| coin_flip | 661 | 0.500 [0.461, 0.537] | 0.2500 [0.2500, 0.2500] | 0.6931 [0.6931, 0.6931] | 0.0507 | - |
| elo_only | 661 | 0.660 [0.623, 0.695] | 0.2157 [0.2030, 0.2294] | 0.6216 [0.5917, 0.6540] | 0.0163 | - |
| higher_elo_pick | 661 | 0.660 [0.623, 0.695] | n/a | n/a | n/a | - |
| old_engine_v1 | 661 | 0.569 [0.531, 0.606] | 0.3172 [0.2907, 0.3436] | 1.1013 [0.9657, 1.2502] | 0.2685 | 0.248 |
| old_engine_v1_posthoc_T | 661 | 0.569 [0.531, 0.606] | 0.2367 [0.2296, 0.2432] | 0.6646 [0.6495, 0.6792] | 0.0414 | - |
| new_engine | 661 | 0.654 [0.616, 0.689] | 0.2152 [0.2029, 0.2283] | 0.6193 [0.5927, 0.6479] | 0.0240 | 0.369 |
| new_engine_T1 | 661 | 0.654 [0.616, 0.689] | 0.2157 [0.2021, 0.2303] | 0.6203 [0.5903, 0.6527] | 0.0366 | 0.366 |

Notes: coin flip accuracy is credited 0.5 per series by definition. higher_elo_pick is a hard pick (accuracy only). old_engine_v1_posthoc_T is a diagnostic, not the shipped old engine. Scoreline acc = modal BO3 scoreline from series_probs vs actual (test base rates: 2-0 0.32, 2-1 0.23, 1-2 0.19, 0-2 0.26).

## Paired Brier differences (test, 95% paired bootstrap; negative = first model better)

- new_engine - elo_only: -0.0005 [-0.0030, +0.0017]; log-loss diff -0.0023
- new_engine - old_engine_v1: -0.1020 [-0.1249, -0.0795]; log-loss diff -0.4821
- old_engine_v1 - elo_only: +0.1015 [+0.0791, +0.1252]; log-loss diff +0.4798
- new_engine - coin_flip: -0.0348 [-0.0471, -0.0217]; log-loss diff -0.0739

## Reliability table: new_engine (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 0 | - | - |
| 0.1-0.2 | 3 | 0.198 | 0.000 |
| 0.2-0.3 | 81 | 0.247 | 0.247 |
| 0.3-0.4 | 82 | 0.354 | 0.366 |
| 0.4-0.5 | 94 | 0.453 | 0.489 |
| 0.5-0.6 | 118 | 0.552 | 0.525 |
| 0.6-0.7 | 107 | 0.650 | 0.682 |
| 0.7-0.8 | 169 | 0.763 | 0.746 |
| 0.8-0.9 | 7 | 0.805 | 1.000 |
| 0.9-1.0 | 0 | - | - |

## Reliability table: old_engine_v1 (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 108 | 0.035 | 0.426 |
| 0.1-0.2 | 66 | 0.142 | 0.545 |
| 0.2-0.3 | 46 | 0.247 | 0.500 |
| 0.3-0.4 | 29 | 0.345 | 0.414 |
| 0.4-0.5 | 39 | 0.449 | 0.538 |
| 0.5-0.6 | 33 | 0.555 | 0.515 |
| 0.6-0.7 | 38 | 0.644 | 0.500 |
| 0.7-0.8 | 46 | 0.752 | 0.478 |
| 0.8-0.9 | 67 | 0.855 | 0.507 |
| 0.9-1.0 | 189 | 0.967 | 0.709 |

## Reliability table: elo_only (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 3 | 0.081 | 0.333 |
| 0.1-0.2 | 22 | 0.154 | 0.136 |
| 0.2-0.3 | 44 | 0.257 | 0.250 |
| 0.3-0.4 | 78 | 0.349 | 0.372 |
| 0.4-0.5 | 113 | 0.448 | 0.442 |
| 0.5-0.6 | 129 | 0.550 | 0.550 |
| 0.6-0.7 | 125 | 0.650 | 0.672 |
| 0.7-0.8 | 72 | 0.753 | 0.736 |
| 0.8-0.9 | 57 | 0.845 | 0.842 |
| 0.9-1.0 | 18 | 0.927 | 0.778 |

## Reliability tiers / bands: new_engine

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 407 | 0.695 | 0.2015 | 25.7 | 0.685 |
| MEDIUM | 145 | 0.621 | 0.2292 | 31.3 | 0.637 |
| LOW | 109 | 0.541 | 0.2475 | 37.0 | 0.606 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 165 | 21.0-25.2 | 0.2000 | 0.685 |
| Q2 | 165 | 25.2-27.7 | 0.1869 | 0.745 |
| Q3 | 165 | 27.7-31.7 | 0.2285 | 0.624 |
| Q4 | 166 | 31.8-48.6 | 0.2451 | 0.560 |

## Reliability tiers / bands: old_engine_v1

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 0 | - | - | - | - |
| MEDIUM | 41 | 0.585 | 0.2947 | 36.7 | 0.739 |
| LOW | 620 | 0.568 | 0.3187 | 56.9 | 0.844 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 165 | 30.6-49.8 | 0.2561 | 0.606 |
| Q2 | 165 | 49.8-58.1 | 0.2938 | 0.576 |
| Q3 | 165 | 58.1-64.0 | 0.3577 | 0.527 |
| Q4 | 166 | 64.0-64.0 | 0.3609 | 0.566 |

## Overconfidence check (outputs >= 90% or <= 10%)

- old_engine_v1 (test): 297 series (44.9%), hit-rate 0.660, mean stated conf 0.967
- new_engine (test): 0 series (0.0%), hit-rate -, mean stated conf -
- elo_only (test): 21 series (3.2%), hit-rate 0.762, mean stated conf 0.926
- old_engine_v1 (all 1659 eval series): 712 (42.9%), hit-rate 0.640, mean stated conf 0.967

## Feature derivation (point-in-time, only series dated strictly before the match day)

- rating: series Elo (init 1500, K=32 x {1: 0.75, 3: 1.0, 5: 1.25} by best-of), rating = 1 + (elo-1500)/2000. PROXY, not HLTV.
- form30/n30/opp_rating30: series win rate, count and mean opponent proxy rating in the 30 days before; keys omitted if no series.
- form5: win rate over the last <=5 series; omitted if none.
- h2h: series wins vs each other in the prior 365 days.
- maps_a/b: per-map [win rate, maps played] over the prior 90 days.
- map_pool: maps with >= 3 plays (all teams) in the prior 60 days, the 7 most recently played (old engine ignores it: hardcoded pool).
- volatility: |wins - Elo-expected wins| / n over last 10 series x 2.5, clamped [0,1]; 0.3 if < 3 series.
- permaban: None (no veto data); roster: stand-in and missing-IGL flags from Liquipedia team-page stand-in tables (data/rosters.py); stakes: none.

