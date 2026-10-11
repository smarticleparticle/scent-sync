#!/usr/bin/env python
"""Build config/goodscents_selected.csv: the GoodScents molecules the free-response scoring uses (the first three name matches of every search term).

The notebook downloaded the three pyrfume-data GoodScents tables from the `main` branch every time; the scoring only ever uses these ~15 rows, so the
subset is committed and the analysis needs no internet.  This script rebuilds it from a pinned pyrfume-data commit (or a local copy) and records the
SHA-256 of the three source files, so the subset can be audited.

    python scripts/make_goodscents_subset.py --ref <commit-sha> --out config/goodscents_selected.csv        # online
    python scripts/make_goodscents_subset.py --dir /path/to/goodscents --out config/goodscents_selected.csv  # local copy
  (a commit id: git ls-remote https://github.com/pyrfume/pyrfume-data.git HEAD)
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import urllib.request

from olfaction_eeg import word_scoring as ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="main")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    hashes = {}
    for name in ("molecules", "behavior", "stimuli"):
        if a.dir:
            data = open(os.path.join(a.dir, f"{name}.csv"), "rb").read()
        else:
            data = urllib.request.urlopen(ws.PYRFUME_RAW.format(ref=a.ref, name=name), timeout=120).read()
        hashes[f"{name}.csv"] = hashlib.sha256(data).hexdigest()
    gs = ws.load_goodscents(a.dir, ref=a.ref)
    with contextlib.redirect_stdout(io.StringIO()):
        found = ws.search_all_datasets_fixed({"goodscents": gs})
    sel = ws.selected_table(found)
    sel.to_csv(a.out, index=False)
    json.dump({"source": "pyrfume-data/goodscents", "ref": a.dir or a.ref, "n_rows_merged": int(len(gs)), "sha256": hashes},
              open(os.path.splitext(a.out)[0] + "_source.json", "w"), indent=2)
    print(f"{len(sel)} molecules -> {a.out}  (merged GoodScents rows: {len(gs)})")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
