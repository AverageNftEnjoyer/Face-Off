# Face-Off
Probabilistic CS2 matchup analysis: win probabilities, map veto simulation, and calibrated confidence intervals for every series

## Layout

| Folder | What's in it |
|---|---|
| `model/` | prediction engine, veto model, rating history |
| `evaluation/` | walk-forward evaluation, weight fitting, leakage checks |
| `collectors/` | Liquipedia / Valve fetchers and parsers |
| `pipeline/` | scheduled refresh jobs |
| `checks/` | data gates and freshness checks |
| `viewer/` | site builder and page template |
| `data/` | the data (no code) |
| `tests/` | `python -m unittest discover -s tests` |

How it all fits together, and a change log: [ARCHITECTURE.md](ARCHITECTURE.md).
