#!/usr/bin/env python
"""Rewrite MANIFEST.sha256 for a deposit folder from the files that are in it now, and give every file the same neutral modification date.

Use it after adding or removing a file by hand (building the deposit again with make_deposit.py is the other way):

    python scripts/refresh_manifest.py --root deposit_final
"""
import argparse
import os

from olfaction_eeg import deposit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    entries = []
    for dp, _, fs in os.walk(a.root):
        for f in fs:
            if f == "MANIFEST.sha256" or f.startswith("."):
                continue
            p = os.path.join(dp, f)
            entries.append((p, os.path.relpath(p, a.root).replace(os.sep, "/")))
    entries.sort(key=lambda e: (e[1].count("/"), e[1]))
    manifest = os.path.join(a.root, "MANIFEST.sha256")
    deposit.write_manifest(entries, manifest)
    deposit.normalise_mtimes(a.root)
    for _, name in entries:
        print(name)
    print(f"\n{len(entries)} files listed in {manifest}; modification times set to a neutral date.")


if __name__ == "__main__":
    main()
