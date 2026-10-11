#!/usr/bin/env python
"""Trial-by-trial comparison of two epochs files (e.g. the original epochs_eeg_respiration.npy vs ..._withtimestamps_v2.npy).

`compare_npy_dicts.py` says THAT two files differ.  This script says HOW: for every subject and condition each trial of
file A is matched to its best-correlated trial in file B and classified as

    exact        same numbers (np.allclose)
    rescaled     same waveform (corr >= 0.9999) but different amplitude/offset   (filtering, referencing, units ...)
    similar      0.9 <= corr < 0.9999                                            (different processing of the same trial?)
    no_match     corr < 0.9                                                      (a trial that does not exist in B)

and B trials that no A trial matches are counted too.  Typical readings:
    A trials all `exact`, B has extra trials          -> A = B with trials removed (quality control done at epoching)
    `rescaled` everywhere                             -> same trials, different preprocessing / reference / units
    `no_match` in A and unmatched in B                -> different trial selection or a shifted event alignment

    python scripts/compare_epochs.py A.npy B.npy --out compare_epochs.csv
Needs both files in memory (about twice the size of the larger one).
"""
import argparse
import sys

import numpy as np
import pandas as pd


def _unit_rows(x):
    X = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def compare_trials(a, b, rtol=1e-6, atol=1e-12):
    """a, b: arrays (n_trials, ...).  Returns a dict of counts and diagnostics."""
    a, b = np.asarray(a), np.asarray(b)
    out = dict(n_A=len(a), n_B=len(b), exact=0, rescaled=0, similar=0, no_match=0, B_unmatched=len(b),
               order_preserved=np.nan, median_corr=np.nan, median_scale=np.nan, median_offset=np.nan)
    if len(a) == 0 or len(b) == 0 or a.shape[1:] != b.shape[1:]:
        out["no_match"] = len(a)
        if len(a) and len(b):
            out["note"] = f"trial shapes differ: {a.shape[1:]} vs {b.shape[1:]}"
        return out
    C = _unit_rows(a) @ _unit_rows(b).T
    best, bestc = C.argmax(axis=1), C.max(axis=1)
    scales, offsets = [], []
    for i, (j, c) in enumerate(zip(best, bestc)):
        if np.allclose(a[i], b[j], rtol=rtol, atol=atol):
            out["exact"] += 1
        elif c >= 0.9999:
            out["rescaled"] += 1
        elif c >= 0.9:
            out["similar"] += 1
        else:
            out["no_match"] += 1
        if c >= 0.9:
            x, y = a[i].ravel().astype(np.float64), b[j].ravel().astype(np.float64)
            s = np.dot(x - x.mean(), y - y.mean()) / max(np.dot(x - x.mean(), x - x.mean()), 1e-300)
            scales.append(s)
            offsets.append(y.mean() - s * x.mean())
    matched_b = {int(j) for j, c in zip(best, bestc) if c >= 0.9}
    out["B_unmatched"] = len(b) - len(matched_b)
    ok = [int(j) for j, c in zip(best, bestc) if c >= 0.9]
    out["order_preserved"] = bool(all(x < y for x, y in zip(ok, ok[1:]))) if len(ok) > 1 else np.nan
    out["median_corr"] = float(np.median(bestc))
    if scales:
        out["median_scale"], out["median_offset"] = float(np.median(scales)), float(np.median(offsets))
    return out


def compare_files(A, B, fields=("EEG", "respiration")):
    rows, only = [], {"A": set(), "B": set()}
    for field in fields:
        if field not in A or field not in B:
            print(f"field {field!r} missing in {'A' if field not in A else 'B'}")
            continue
        for cond in sorted(set(A[field]) | set(B[field])):
            sa, sb = A[field].get(cond, {}), B[field].get(cond, {})
            only["A"] |= set(sa) - set(sb)
            only["B"] |= set(sb) - set(sa)
            for subj in sorted(set(sa) & set(sb)):
                rows.append({"field": field, "condition": cond, "subject": subj, **compare_trials(sa[subj], sb[subj])})
    return pd.DataFrame(rows), only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A")
    ap.add_argument("B")
    ap.add_argument("--out", default="compare_epochs.csv")
    a = ap.parse_args()
    A = np.load(a.A, allow_pickle=True).item()
    B = np.load(a.B, allow_pickle=True).item()
    df, only = compare_files(A, B)
    df.to_csv(a.out, index=False)
    print(f"\nsubjects only in A: {sorted(only['A'])}   only in B: {sorted(only['B'])}")
    for field, g in df.groupby("field"):
        tot = g[["n_A", "n_B", "exact", "rescaled", "similar", "no_match", "B_unmatched"]].sum()
        print(f"\n=== {field}: {len(g)} subject x condition pairs in both files ===")
        print(f"trials in A: {int(tot.n_A)}   trials in B: {int(tot.n_B)}")
        print(f"A trials -> exact {int(tot.exact)} | rescaled {int(tot.rescaled)} | similar {int(tot.similar)} | no match {int(tot.no_match)}")
        print(f"B trials with no matching A trial: {int(tot.B_unmatched)}")
        same_n = g[g.n_A == g.n_B]
        print(f"pairs with equal trial counts: {len(same_n)}, of which all-exact: {int((same_n.exact == same_n.n_A).sum())}")
        if g.median_scale.notna().any():
            print(f"median scale / offset of matched non-exact trials: {g.median_scale.median():.4g} / {g.median_offset.median():.3g}")
        print("order preserved in", int((g.order_preserved == True).sum()), "of", int(g.order_preserved.notna().sum()), "pairs")  # noqa: E712
    print(f"\nfull table: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
