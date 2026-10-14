"""Where everything lives (the one place the folder layout is written down in code).

Every script and test starts with the same two lines:

    sys.path.insert(0, <repo root>)
    import paths  # noqa: E402

Importing this puts each code folder on sys.path, so modules import each other
by their flat names (`import backtest`, `import lpfetch`, `import events`) from
anywhere in the repo. A new code folder goes in CODE_DIRS (and in
ARCHITECTURE.md; tests/test_architecture.py checks both).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")          # inputs, caches, committed outputs: no code
RAW = os.path.join(DATA, "raw")            # fetched pages, standings files, media
VIEWER = os.path.join(ROOT, "viewer")      # site builder + its template and media

CODE_DIRS = (
    "model",        # predictor, veto, rating/feature history
    "evaluation",   # walk-forward, weight fitting, leakage checks
    "collectors",   # Liquipedia / Valve fetchers and parsers -> data/*.json
    "pipeline",     # scheduled refresh jobs and data maintenance
    "checks",       # data gates and freshness / consistency checks
    "viewer",       # bundle + static page builder
)

for _d in reversed(CODE_DIRS):
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
