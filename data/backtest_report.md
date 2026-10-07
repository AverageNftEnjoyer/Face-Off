# CS2 engine backtest report

Fan analytics only. Real Liquipedia results (data/SOURCES.md). Ratings are a point-in-time **Elo proxy**, not HLTV ratings.

## Data

- series: 5551 ({'1': 562, '3': 4921, '5': 68}) from 2023-10-16 to 2026-10-07, 419 teams
- BO3 series: 4921; excluded (either team < 5 prior series): 1278
- evaluated: 3643 -> train 2187 (2023-10-22..2026-04-25), test 1456 (2026-04-26..2026-10-06)
- |rating gap| quantiles (proxy scale): q25=0.014, q50=0.030, q75=0.055, q90=0.087

## Temperature (fitted on train only)

- T = **0.911** (engine CONFIG['temperature'] fitted on total_logodds (train))
- train log-loss: T=1 0.6454 -> fitted 0.6450

## Test-set metrics

| model | n | accuracy [95% Wilson] | Brier [95% boot] | log-loss [95% boot] | ECE10 | scoreline acc |
|---|---|---|---|---|---|---|
| coin_flip | 1456 | 0.500 [0.474, 0.526] | 0.2500 [0.2500, 0.2500] | 0.6931 [0.6931, 0.6931] | 0.0721 | - |
| elo_only | 1456 | 0.598 [0.572, 0.622] | 0.2334 [0.2276, 0.2395] | 0.6589 [0.6464, 0.6720] | 0.0447 | - |
| higher_elo_pick | 1456 | 0.598 [0.572, 0.622] | n/a | n/a | n/a | - |
| old_engine_v1 | 1456 | 0.566 [0.540, 0.591] | 0.3324 [0.3135, 0.3518] | 1.4080 [1.2426, 1.5826] | 0.2731 | 0.267 |
| old_engine_v1_posthoc_T | 1456 | 0.566 [0.540, 0.591] | 0.2457 [0.2409, 0.2506] | 0.6868 [0.6757, 0.6983] | 0.0649 | - |
| new_engine | 1456 | 0.597 [0.571, 0.622] | 0.2335 [0.2270, 0.2404] | 0.6591 [0.6452, 0.6739] | 0.0472 | 0.341 |
| new_engine_T1 | 1456 | 0.597 [0.571, 0.622] | 0.2340 [0.2270, 0.2414] | 0.6603 [0.6451, 0.6764] | 0.0495 | 0.343 |

Notes: coin flip accuracy is credited 0.5 per series by definition. higher_elo_pick is a hard pick (accuracy only). old_engine_v1_posthoc_T is a diagnostic, not the shipped old engine. Scoreline acc = modal BO3 scoreline from series_probs vs actual (test base rates: 2-0 0.34, 2-1 0.23, 1-2 0.20, 0-2 0.23).

## Paired Brier differences (test, 95% paired bootstrap; negative = first model better)

- new_engine - elo_only: +0.0001 [-0.0011, +0.0013]; log-loss diff +0.0002
- new_engine - old_engine_v1: -0.0989 [-0.1148, -0.0832]; log-loss diff -0.7489
- old_engine_v1 - elo_only: +0.0990 [+0.0827, +0.1155]; log-loss diff +0.7491
- new_engine - coin_flip: -0.0165 [-0.0230, -0.0096]; log-loss diff -0.0340

## Reliability table: new_engine (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 0 | - | - |
| 0.1-0.2 | 6 | 0.194 | 0.500 |
| 0.2-0.3 | 73 | 0.253 | 0.260 |
| 0.3-0.4 | 187 | 0.358 | 0.401 |
| 0.4-0.5 | 326 | 0.456 | 0.555 |
| 0.5-0.6 | 382 | 0.551 | 0.586 |
| 0.6-0.7 | 302 | 0.646 | 0.659 |
| 0.7-0.8 | 164 | 0.753 | 0.713 |
| 0.8-0.9 | 16 | 0.805 | 0.938 |
| 0.9-1.0 | 0 | - | - |

## Reliability table: old_engine_v1 (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 293 | 0.031 | 0.498 |
| 0.1-0.2 | 120 | 0.148 | 0.567 |
| 0.2-0.3 | 79 | 0.256 | 0.532 |
| 0.3-0.4 | 68 | 0.344 | 0.397 |
| 0.4-0.5 | 92 | 0.457 | 0.522 |
| 0.5-0.6 | 81 | 0.547 | 0.568 |
| 0.6-0.7 | 91 | 0.648 | 0.473 |
| 0.7-0.8 | 98 | 0.749 | 0.602 |
| 0.8-0.9 | 136 | 0.850 | 0.625 |
| 0.9-1.0 | 398 | 0.969 | 0.676 |

## Reliability table: elo_only (P(team_a) bins)

| bin | n | mean pred | actual |
|---|---|---|---|
| 0.0-0.1 | 1 | 0.063 | 0.000 |
| 0.1-0.2 | 5 | 0.180 | 0.200 |
| 0.2-0.3 | 42 | 0.254 | 0.238 |
| 0.3-0.4 | 165 | 0.360 | 0.406 |
| 0.4-0.5 | 370 | 0.455 | 0.527 |
| 0.5-0.6 | 455 | 0.548 | 0.589 |
| 0.6-0.7 | 293 | 0.641 | 0.669 |
| 0.7-0.8 | 99 | 0.739 | 0.758 |
| 0.8-0.9 | 21 | 0.851 | 0.810 |
| 0.9-1.0 | 5 | 0.928 | 0.800 |

## Reliability tiers / bands: new_engine

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 672 | 0.610 | 0.2283 | 26.7 | 0.637 |
| MEDIUM | 386 | 0.578 | 0.2430 | 31.4 | 0.609 |
| LOW | 398 | 0.593 | 0.2333 | 38.1 | 0.591 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 364 | 20.9-27.0 | 0.2268 | 0.626 |
| Q2 | 364 | 27.0-29.9 | 0.2283 | 0.602 |
| Q3 | 364 | 29.9-34.1 | 0.2466 | 0.569 |
| Q4 | 364 | 34.1-56.7 | 0.2324 | 0.591 |

## Reliability tiers / bands: old_engine_v1

| tier | n | accuracy | Brier | mean band width pp | mean conf |
|---|---|---|---|---|---|
| HIGH | 0 | - | - | - | - |
| MEDIUM | 111 | 0.532 | 0.2900 | 35.6 | 0.678 |
| LOW | 1345 | 0.569 | 0.3359 | 56.8 | 0.849 |

Band-width quartiles (does a wider band mean a harder-to-call match?):

| quartile | n | width pp range | Brier | accuracy |
|---|---|---|---|---|
| Q1 | 364 | 25.5-48.6 | 0.2733 | 0.585 |
| Q2 | 364 | 48.6-57.5 | 0.3095 | 0.588 |
| Q3 | 364 | 57.5-64.0 | 0.3524 | 0.555 |
| Q4 | 364 | 64.0-64.0 | 0.3944 | 0.536 |

## Overconfidence check (outputs >= 90% or <= 10%)

- old_engine_v1 (test): 691 series (47.5%), hit-rate 0.602, mean stated conf 0.969
- new_engine (test): 0 series (0.0%), hit-rate -, mean stated conf -
- elo_only (test): 6 series (0.4%), hit-rate 0.833, mean stated conf 0.929
- old_engine_v1 (all 3643 eval series): 1703 (46.7%), hit-rate 0.615, mean stated conf 0.968

## Feature derivation (point-in-time, only series dated strictly before the match day)

- rating: series Elo (init 1500, K=32 x {1: 0.75, 3: 1.0, 5: 1.25} by best-of), rating = 1 + (elo-1500)/2000. PROXY, not HLTV.
- form30/n30/opp_rating30: series win rate, count and mean opponent proxy rating in the 30 days before; keys omitted if no series.
- form5: win rate over the last <=5 series; omitted if none.
- h2h: series wins vs each other in the prior 365 days.
- maps_a/b: per-map [win rate, maps played] over the prior 90 days.
- map_pool: maps with >= 3 plays (all teams) in the prior 60 days, the 7 most recently played (old engine ignores it: hardcoded pool).
- volatility: |wins - Elo-expected wins| / n over last 10 series x 2.5, clamped [0,1]; 0.3 if < 3 series.
- permaban: None (no veto data); roster: stand-in and missing-IGL flags from Liquipedia team-page stand-in tables (data/rosters.py); stakes: none.

