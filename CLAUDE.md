# Working in this repo

Read `ARCHITECTURE.md` first: it is the map of folders, modules, data files and
workflows.

## Keep ARCHITECTURE.md current (every change)

In the same change as the code:

1. **Module map**: add, move, rename or remove the file's row when a module
   is added, moved, renamed or deleted, or when its role changes.
2. **Data files / Data flow / Automation**: update when a data file, a
   pipeline step or a workflow step is added, removed or changes owner.
3. **Change log**: add one dated line at the top (`- **YYYY-MM-DD**: ...`) for
   anything that touches structure, data flow, a data file, a workflow or a
   visible behaviour of the site. Pure refactors inside one module and test-only
   changes need no entry.

`python -m unittest tests.test_architecture` checks the module map and layout.

## Conventions

- Code lives only in the folders listed in `paths.CODE_DIRS`; `data/` holds
  data only. Module names are unique across folders (one flat import
  namespace). New scripts and tests start with the two-line `paths` bootstrap
  (see ARCHITECTURE.md, "Imports").
- Data paths come from `paths.DATA` / `paths.RAW`, never from a module's own location.
- Stdlib only (Vercel installs nothing).
- Before handing work back: `python -m unittest discover -s tests`, and for
  site changes `python viewer/build_bundle.py viewer/data-bundle.json &&
  python viewer/build_viewer.py out/index.html`.
