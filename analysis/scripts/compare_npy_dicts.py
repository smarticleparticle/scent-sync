#!/usr/bin/env python
"""Compare two pickled-dict .npy files (epochs files, ERSP result files, ...) and say whether they are identical --
and if not, where and by how much they differ.

    python scripts/compare_npy_dicts.py A.npy B.npy                  # exact comparison
    python scripts/compare_npy_dicts.py A.npy B.npy --tol 1e-9       # differences <= 1e-9 (absolute and relative) count as equal

Use --tol 1e-9 for ERSP files made in different software environments (numpy / scipy / PyWavelets versions change the last
digits).  The summary always reports the LARGEST difference found, so noise (1e-13) is easy to tell from a real difference.
Keys present in only one file are reported separately (e.g. a 'timestamps' entry that only the v2 epochs file has).
"""
import sys

import numpy as np


def compare(a, b, path="", out=None, tol=0.0, stats=None):
    out = [] if out is None else out
    stats = {"max_abs": 0.0, "n_arrays": 0, "n_equal_within_tol": 0} if stats is None else stats
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) - set(b), key=str):
            out.append(f"only in A: {path}/{k}")
        for k in sorted(set(b) - set(a), key=str):
            out.append(f"only in B: {path}/{k}")
        for k in sorted(set(a) & set(b), key=str):
            compare(a[k], b[k], f"{path}/{k}", out, tol, stats)
    else:
        a, b = np.asarray(a), np.asarray(b)
        stats["n_arrays"] += 1
        if a.shape != b.shape:
            out.append(f"shape differs at {path}: {a.shape} vs {b.shape}")
        elif a.dtype.kind in "fc" or b.dtype.kind in "fc":
            with np.errstate(invalid="ignore"):
                d = np.nanmax(np.abs(a.astype(float) - b.astype(float))) if a.size else 0.0
            stats["max_abs"] = max(stats["max_abs"], float(d))
            if np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True):
                stats["n_equal_within_tol"] += 1
            else:
                out.append(f"values differ at {path} (max abs diff {d:.3g})")
        elif not np.array_equal(a, b):
            out.append(f"values differ at {path}")
        else:
            stats["n_equal_within_tol"] += 1
    return out


def main(argv):
    args = [x for x in argv if not x.startswith("--")]
    tol = 0.0
    if "--tol" in argv:
        tol = float(argv[argv.index("--tol") + 1])
        args = [x for x in args if x != argv[argv.index("--tol") + 1]]
    if len(args) < 2:
        sys.exit(__doc__)
    A = np.load(args[0], allow_pickle=True).item()
    B = np.load(args[1], allow_pickle=True).item()
    stats = {"max_abs": 0.0, "n_arrays": 0, "n_equal_within_tol": 0}
    diffs = compare(A, B, tol=tol, stats=stats)
    print(f"compared {stats['n_arrays']} arrays; largest absolute difference among numeric arrays: {stats['max_abs']:.3g}")
    if not diffs:
        print("IDENTICAL" + (f" within tolerance {tol:g}" if tol else " (same keys, shapes and values)"))
    else:
        print(f"{len(diffs)} difference(s); first 25:")
        print("\n".join(diffs[:25]))


if __name__ == "__main__":
    main(sys.argv[1:])
