#!/usr/bin/env python3
"""
page_smoke.py -- refuse to ship a broken page. Run on the built page right
after viewer/build_viewer.py, in every place a page is built: the Vercel
build (vercel.json buildCommand), the refresh workflow's site-build check,
ci.yml and the pre-push hook. A non-zero exit fails that build, so Vercel
keeps serving the last good deployment instead of a blank or broken site.

Checks:
  * the page exists and is a plausible size;
  * every build placeholder was filled (no /*__X__*/ left) and the data
    constants the page cannot start without are not null (DATA, FIXTURES,
    RESULTS, META), and the page's bundle version is set;
  * the page script parses: `node --check` on the inline script. This is
    the check that catches a typo or a duplicate declaration, which would
    otherwise ship a blank page. Node is present on Vercel and GitHub's
    runners; where it is missing the check is skipped with a warning, and
    in CI / on Vercel a missing node is itself a failure.

USAGE (repo root):  python checks/page_smoke.py public/index.html
Stdlib only.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

MIN_BYTES = 1_000_000          # the real page is ~13 MB; anything tiny is a failed build
REQUIRED = ("DATA", "FIXTURES", "RESULTS", "META")


def check(path):
    """-> list of problems (empty = ship it)."""
    if not os.path.exists(path):
        return [f"{path} does not exist"]
    with open(path, encoding="utf-8") as f:
        html = f.read()
    errs = []
    if len(html.encode("utf-8")) < MIN_BYTES:
        errs.append(f"page is only {len(html) // 1024} KB")
    left = sorted(set(re.findall(r"/\*__([A-Z]+)__\*/", html)))
    if left:
        errs.append(f"unfilled build placeholders: {left}")
    for name in REQUIRED:
        m = re.search(rf"const {name} = (.)", html)
        if not m:
            errs.append(f"const {name} is missing")
        elif m.group(1) == "n" and re.search(rf"const {name} = null\b", html):
            errs.append(f"const {name} is null (the build did not inline it)")
    if not re.search(r"const PAGE_BUNDLE_VERSION = \d+;", html):
        errs.append("PAGE_BUNDLE_VERSION is missing")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    if not scripts:
        errs.append("no inline script")
        return errs
    node = shutil.which("node")
    if not node:
        msg = "node not found: the page script was not syntax-checked"
        if os.environ.get("CI") or os.environ.get("VERCEL"):
            errs.append(msg)
        else:
            print(f"page_smoke: WARNING {msg}", file=sys.stderr)
        return errs
    with tempfile.TemporaryDirectory() as tmp:
        for i, js in enumerate(scripts):
            p = os.path.join(tmp, f"s{i}.js")
            with open(p, "w", encoding="utf-8") as f:
                f.write(js)
            r = subprocess.run([node, "--check", p], capture_output=True, text=True)
            if r.returncode:
                detail = (r.stderr or r.stdout).strip().replace(tmp, "<page>")
                errs.append(f"page script {i} does not parse:\n{detail[:1500]}")
    return errs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else os.path.join("public", "index.html")
    errs = check(path)
    if errs:
        print(f"page_smoke: FAIL {path}", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"page_smoke: OK {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
