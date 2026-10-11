#!/usr/bin/env python
"""HOW do two ERSP result files differ?  (e.g. the paper's phase_aligned_...trials.npy vs the pipeline's output)

`compare_npy_dicts.py` says THAT they differ.  This script characterises the difference so the cause can be narrowed down:

  * keys: subjects / trial indices present in only one file  (different trial selection / quality control?)
  * for a sample of trial arrays present in both:
        correlation          ~1.00  same pattern  -> scaling / normalisation / constant offset
                             0.8-0.99 same data but somewhat different channel set or processing
                             < 0.5  different data or misaligned axes
        mean difference      a constant offset in log10 power (a multiplicative factor in power)
        std ratio            B/A spread of the values (1 = same dynamic range)
        profile over frequency / over phase of the mean difference   (frequency-dependent -> wavelet / filtering difference;
                                                                     phase-dependent -> different respiratory-phase alignment)
        NaN pattern          same empty phase bins?

    python scripts/diagnose_ersp_mismatch.py PAPER_FILE.npy PIPELINE_OUTPUT.npy [--max-arrays 300]
"""
import argparse
import sys

import numpy as np


def iter_arrays(D):
    for cond, subjects in D.items():
        if cond in ("freqs", "x") or not isinstance(subjects, dict):
            continue
        for subj, trials in subjects.items():
            for tk, arr in trials.items():
                yield cond, subj, tk, np.asarray(arr)


def key_report(A, B):
    ka = {(c, s, t) for c, s, t, _ in iter_arrays(A)}
    kb = {(c, s, t) for c, s, t, _ in iter_arrays(B)}
    subj_a = {(c, s) for c, s, _ in ka}
    subj_b = {(c, s) for c, s, _ in kb}
    return dict(n_A=len(ka), n_B=len(kb), common=len(ka & kb), only_A=sorted(ka - kb), only_B=sorted(kb - ka),
                subj_only_A=sorted(subj_a - subj_b), subj_only_B=sorted(subj_b - subj_a))


def array_stats(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 3:
        return None
    x, y = a[m], b[m]
    d = y - x
    return dict(corr=float(np.corrcoef(x, y)[0, 1]), mean_diff=float(d.mean()), std_diff=float(d.std()),
                std_ratio=float(y.std() / (x.std() + 1e-300)), nan_same=bool(np.array_equal(np.isnan(a), np.isnan(b))))


def verdict(med, freq_range, n_only, corr_fc):
    msgs = []
    if n_only:
        msgs.append(f"TRIAL SELECTION differs: {n_only} trial/subject keys exist in only one file (quality control or epochs differ)")
    if freq_range > 0.05 and corr_fc > 0.95 and med["corr"] < corr_fc - 0.1:
        msgs.append(f"FREQUENCY-DEPENDENT OFFSET: the correlation is only {med['corr']:.3f}, but after removing a per-frequency offset "
                    f"(range {freq_range:.3f} log10 units across frequencies) it is {corr_fc:.3f}: the pattern is intact and differs by a "
                    "frequency-dependent factor -> wavelet definition / normalisation, or EEG filtering, differs")
        return msgs
    if med["corr"] > 0.99 and abs(med["mean_diff"]) > 0.02:
        msgs.append(f"CONSTANT OFFSET: same pattern (corr {med['corr']:.3f}) but B is {med['mean_diff']:+.3f} log10 units "
                    f"(= x{10 ** med['mean_diff']:.2f} in power): a normalisation / channel-averaging / units difference")
    elif med["corr"] > 0.99:
        msgs.append("NUMERICALLY CLOSE: same pattern and offset; remaining differences are small")
    elif med["corr"] > 0.8:
        msgs.append(f"SAME PATTERN, DIFFERENT DETAILS (corr {med['corr']:.3f}): typical of a different CHANNEL SET (the ROI is the "
                    "intersection over all info files) or a small processing difference")
    elif med["corr"] > 0.5:
        msgs.append(f"PARTLY RELATED (corr {med['corr']:.3f}): a much different channel set, different EEG preprocessing, or a different "
                    "respiratory-phase alignment")
    else:
        msgs.append(f"UNRELATED (corr {med['corr']:.3f}): different EEG data, or the frequency/phase axes are not the same")
    if abs(med["std_ratio"] - 1) > 0.1:
        msgs.append(f"DYNAMIC RANGE differs (spread of B is {med['std_ratio']:.2f}x that of A)")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A", help="reference file (e.g. the paper's ERSP file)")
    ap.add_argument("B", help="file to examine (e.g. pipeline output)")
    ap.add_argument("--max-arrays", type=int, default=300)
    a = ap.parse_args()
    A = np.load(a.A, allow_pickle=True).item()
    B = np.load(a.B, allow_pickle=True).item()

    print("freqs equal:", np.allclose(A["freqs"], B["freqs"]), "| x equal:", np.allclose(A["x"], B["x"]) if len(A["x"]) == len(B["x"]) else f"different lengths {len(A['x'])} vs {len(B['x'])}")
    kr = key_report(A, B)
    print(f"\ntrial arrays: A {kr['n_A']}, B {kr['n_B']}, in both {kr['common']}")
    print(f"subject x condition blocks only in A: {kr['subj_only_A'][:10]}  only in B: {kr['subj_only_B'][:10]}")
    print(f"trial keys only in A: {len(kr['only_A'])} {kr['only_A'][:6]}   only in B: {len(kr['only_B'])} {kr['only_B'][:6]}")

    rng = np.random.default_rng(0)
    pairs = []
    Bd = {(c, s, t): arr for c, s, t, arr in iter_arrays(B)}
    for c, s, t, arr in iter_arrays(A):
        if (c, s, t) in Bd and Bd[(c, s, t)].shape == arr.shape:
            pairs.append((c, s, t, arr, Bd[(c, s, t)]))
    if not pairs:
        print("No comparable arrays.")
        return
    idx = rng.permutation(len(pairs))[: a.max_arrays]
    stats, fprof, pprof = [], [], []
    for i in idx:
        c, s, t, x, y = pairs[i]
        st = array_stats(x, y)
        if st is None:
            continue
        stats.append(st)
        d = y - x
        fprof.append(np.nanmean(d, axis=1))
        pprof.append(np.nanmean(d, axis=0))
    med = {k: float(np.median([s[k] for s in stats])) for k in ("corr", "mean_diff", "std_diff", "std_ratio")}
    print(f"\n=== {len(stats)} sampled arrays: medians (B relative to A) ===")
    print(f"correlation {med['corr']:.4f} | mean difference {med['mean_diff']:+.4f} log10 units | std of difference {med['std_diff']:.4f} | "
          f"std ratio B/A {med['std_ratio']:.3f} | NaN pattern identical in {np.mean([s['nan_same'] for s in stats]):.0%}")
    print(f"correlation: 10th / 90th percentile {np.percentile([s['corr'] for s in stats], 10):.3f} / {np.percentile([s['corr'] for s in stats], 90):.3f}")
    fprof = np.nanmean(fprof, axis=0)
    freqs = np.asarray(A["freqs"])
    sel = np.linspace(0, len(freqs) - 1, 8).astype(int)
    print("\nmean difference by frequency:   " + "  ".join(f"{freqs[i]:5.1f}Hz {fprof[i]:+.3f}" for i in sel))
    pprof = np.nanmean(pprof, axis=0)
    xs = np.asarray(A["x"])
    sel = np.linspace(0, len(xs) - 1, 8).astype(int)
    print("mean difference by phase/time:  " + "  ".join(f"{xs[i]:6.1f} {pprof[i]:+.3f}" for i in sel))
    corr_fc = []
    for i in idx:
        c, s_, t, x_, y_ = pairs[i]
        yc = y_ - fprof[:, None]
        m = ~np.isnan(x_) & ~np.isnan(yc)
        if m.sum() > 3:
            corr_fc.append(float(np.corrcoef(x_[m], yc[m])[0, 1]))
    corr_fc = float(np.median(corr_fc))
    print(f"correlation after removing the mean per-frequency offset: {corr_fc:.4f}")
    print("\nREADING:")
    for m in verdict(med, float(np.nanmax(fprof) - np.nanmin(fprof)), len(kr["only_A"]) + len(kr["only_B"]), corr_fc):
        print("  -", m)


if __name__ == "__main__":
    sys.exit(main())
