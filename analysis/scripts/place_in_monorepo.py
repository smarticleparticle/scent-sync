#!/usr/bin/env python
"""Copy this analysis repository into a subfolder of a larger repository (e.g. the olfactometer repo with firmware/ and hardware/), keeping the rest untouched.

    python scripts/place_in_monorepo.py --target ~/publish/scent-sync --subdir analysis            # dry run
    python scripts/place_in_monorepo.py --target ~/publish/scent-sync --subdir analysis --apply

What goes where:
  <target>/<subdir>/...        everything (code, config, tests, docs, environment.yml, .gitignore) except the files below
  <target>/CITATION.cff        so GitHub's "Cite this repository" and Zenodo find it at the top level
  <subdir>/LICENSE             the license of the analysis code stays inside its folder; the top-level LICENSE (which also covers firmware / hardware) is NOT touched,
                               unless you pass --license-at-root (the existing one is then saved as LICENSE.previous)
  <target>/.github/workflows/analysis-tests.yml    the CI workflow, with paths pointing into <subdir>
Not copied: NOTES_FOR_AUTHORS.md, data and result folders, caches.  Stops if <target> is not a git repository or has uncommitted changes.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", ".snakemake", ".pytest_cache", "deposit", "data_raw", ".ipynb_checkpoints"}
SKIP_PREFIXES = ("results", "figures")
SKIP_FILES = {"NOTES_FOR_AUTHORS.md", ".DS_Store"}
ROOT_FILES = {"CITATION.cff"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--subdir", default="analysis")
    ap.add_argument("--license-at-root", action="store_true", help="also put LICENSE at the top level of the target (replaces its LICENSE; the old one is kept as LICENSE.previous)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root_files = ROOT_FILES | ({"LICENSE"} if a.license_at_root else set())
    target = os.path.abspath(os.path.expanduser(a.target))
    if not os.path.isdir(os.path.join(target, ".git")):
        sys.exit(f"{target} is not a git repository")
    dirty = subprocess.run(["git", "-C", target, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("The target repository has uncommitted changes; commit or stash them first:\n" + dirty)
    dest = os.path.join(target, a.subdir)

    plan = []                                        # (src, dst)
    for dp, dns, fns in os.walk(HERE):
        rel_dir = os.path.relpath(dp, HERE)
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not (rel_dir == "." and d.startswith(SKIP_PREFIXES))
                  and not (rel_dir == "." and d == ".github")
                  and not (rel_dir == "tests" and d == "data")
                  and not (rel_dir == os.path.join("data") and d == "raw")]
        for f in fns:
            if f in SKIP_FILES or f.endswith((".pyc",)):
                continue
            src = os.path.join(dp, f)
            rel = os.path.normpath(os.path.join(rel_dir, f))
            if rel_dir == "." and f in root_files:
                continue
            if rel.startswith(os.path.join("data", "preprocessed") + os.sep) and f != ".gitkeep":
                continue
            plan.append((src, os.path.join(dest, rel)))
    root_plan = []
    for f in sorted(root_files):
        src = os.path.join(HERE, f)
        if os.path.exists(src):
            root_plan.append((src, os.path.join(target, f)))
    wf_src = os.path.join(HERE, ".github", "workflows", "tests.yml")
    workflow = None
    if os.path.exists(wf_src):
        text = open(wf_src, encoding="utf-8").read()
        text = text.replace("environment-file: environment.yml", f"environment-file: {a.subdir}/environment.yml")
        text = text.replace("        shell: bash -el {0}\n", f"        shell: bash -el {{0}}\n        working-directory: {a.subdir}\n")
        workflow = (os.path.join(target, ".github", "workflows", "analysis-tests.yml"), text)

    overwrite = [d for _, d in plan if os.path.exists(d)]
    print(f"{len(plan)} files -> {dest}")
    for _, d in root_plan:
        print(f"top-level: {d}" + ("   (existing file kept as .previous)" if os.path.exists(d) else ""))
    if workflow:
        print(f"workflow:  {workflow[0]}")
    if overwrite:
        print(f"\nThese {len(overwrite)} existing files in the target will be replaced:")
        for d in overwrite:
            print("   ", os.path.relpath(d, target))
    if not a.apply:
        print("\nDRY RUN: add --apply to copy.")
        return
    for src, dst in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    for src, dst in root_plan:
        if os.path.exists(dst):
            shutil.copy2(dst, dst + ".previous")
        shutil.copy2(src, dst)
    if workflow:
        os.makedirs(os.path.dirname(workflow[0]), exist_ok=True)
        open(workflow[0], "w", encoding="utf-8").write(workflow[1])
    print("\nDone. Review with:  git -C", target, "status")


if __name__ == "__main__":
    main()
