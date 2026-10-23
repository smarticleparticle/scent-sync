#!/usr/bin/env python
"""WHY do the EEG arrays of two epochs files differ even though their respiration epochs match?

Trials are paired through the respiration signal (which agrees between the original and v2 file); for each paired trial the
EEG of file A is then compared with the EEG of file B in four ways:

  zero-lag corr        per-channel correlation at the same time points             (high = same signal)
  best lag             cross-correlation peak within +-max_lag samples             (nonzero = EEG epochs start at a different sample)
  avg-ref corr         per-channel correlation after re-referencing both to the average of all channels   (high here but low above = different reference)
  channel permutation  fraction of channels whose best-matching channel in B has the same index          (low = channel order differs)
  std ratio            median per-channel std(B)/std(A)                            (!= 1 = different filtering / scaling / units)

    python scripts/diagnose_eeg_mismatch.py A.npy B.npy --subjects 22 23 --conditions water citrus
"""
import argparse
import sys

import numpy as np


def _unit_rows(x):
    X = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def pair_by_respiration(resp_a, resp_b, min_corr=0.999):
    """[(i, j)] pairs of trials whose respiration traces correlate >= min_corr."""
    C = _unit_rows(resp_a) @ _unit_rows(resp_b).T
    return [(i, int(C[i].argmax())) for i in range(len(resp_a)) if C[i].max() >= min_corr]


def best_lag(x, y, max_lag):
    """Lag k in [-max_lag, max_lag] maximising sum_t x[t+k] y[t]  (k > 0: x is y delayed by k samples)."""
    x, y = x - x.mean(), y - y.mean()
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(y, nfft)), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = cc[lags % nfft] / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-300)
    k = int(np.argmax(vals))
    return int(lags[k]), float(vals[k])


def channel_corr(x, y):
    xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    return (xc * yc).sum(axis=1) / (np.linalg.norm(xc, axis=1) * np.linalg.norm(yc, axis=1) + 1e-300)


def diagnose_pair(x, y, max_lag=500, n_lag_channels=6):
    """x, y: (channels, time) EEG of the same trial from file A and B."""
    out = {}
    out["zero_lag_corr"] = float(np.median(channel_corr(x, y)))
    x_car, y_car = x - x.mean(axis=0, keepdims=True), y - y.mean(axis=0, keepdims=True)
    out["avg_ref_corr"] = float(np.median(channel_corr(x_car, y_car)))
    idx = np.linspace(0, x.shape[0] - 1, n_lag_channels).astype(int)
    lags = [best_lag(x[c], y[c], max_lag) for c in idx]
    out["best_lag"] = int(np.median([l for l, _ in lags]))
    out["lag_peak_corr"] = float(np.median([v for _, v in lags]))
    xs = (x - x.mean(axis=1, keepdims=True)) / (np.linalg.norm(x - x.mean(axis=1, keepdims=True), axis=1, keepdims=True) + 1e-300)
    ys = (y - y.mean(axis=1, keepdims=True)) / (np.linalg.norm(y - y.mean(axis=1, keepdims=True), axis=1, keepdims=True) + 1e-300)
    M = xs @ ys.T                                           # channel x channel correlation
    out["same_channel_is_best"] = float(np.mean(M.argmax(axis=1) == np.arange(x.shape[0])))
    out["best_channel_corr"] = float(np.median(M.max(axis=1)))
    out["std_ratio"] = float(np.median(y.std(axis=1) / (x.std(axis=1) + 1e-300)))
    return out


def verdict(d):
    msgs = []
    scale_off = abs(np.log10(d["std_ratio"] + 1e-300)) > 0.05
    if d["zero_lag_corr"] > 0.95:
        msgs.append("same waveform at zero lag (correlation > 0.95)" + (" but at a different amplitude" if scale_off else ": the EEG agrees"))
    else:
        shift = -d["best_lag"]                       # best_lag > 0 means A equals B delayed; express it as what happened to B
        if shift != 0 and d["lag_peak_corr"] > 0.9:
            how = "delayed" if shift > 0 else "advanced"
            msgs.append(f"TIME SHIFT: B's EEG is A's {how} by {abs(shift)} samples ({abs(shift) / 250:.3f} s at 250 Hz): the epochs start at different samples")
        if d["avg_ref_corr"] > 0.9:
            msgs.append("REFERENCE: matches after average re-referencing (the two files use different references)")
        if d["same_channel_is_best"] < 0.5 and d["best_channel_corr"] > 0.9:
            msgs.append("CHANNEL ORDER: channels match, but not at the same index (channel order or selection differs)")
        if not msgs:
            msgs.append("NO SIMPLE EXPLANATION: not shifted, not a reference change, not a channel permutation -> different EEG "
                        "preprocessing / a different source recording")
    if scale_off:
        msgs.append(f"SCALE: B has {d['std_ratio']:.3g}x the amplitude of A (units, filtering or artefact removal differ)")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A")
    ap.add_argument("B")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: the first 3 subjects in both files")
    ap.add_argument("--conditions", nargs="*", default=["water", "citrus"])
    ap.add_argument("--max-trials", type=int, default=4, help="paired trials examined per subject x condition")
    ap.add_argument("--max-lag", type=int, default=500)
    a = ap.parse_args()
    A = np.load(a.A, allow_pickle=True).item()
    B = np.load(a.B, allow_pickle=True).item()
    common = sorted(set(A["EEG"][a.conditions[0]]) & set(B["EEG"][a.conditions[0]]))
    subjects = a.subjects or common[:3]
    allres = []
    for cond in a.conditions:
        for subj in subjects:
            pairs = pair_by_respiration(A["respiration"][cond][subj], B["respiration"][cond][subj])[: a.max_trials]
            print(f"\n{cond} / subject {subj}: {len(pairs)} trial pairs matched through respiration")
            for i, j in pairs:
                d = diagnose_pair(np.asarray(A["EEG"][cond][subj][i], float), np.asarray(B["EEG"][cond][subj][j], float), a.max_lag)
                allres.append(d)
                print(f"  A trial {i} <-> B trial {j}: zero-lag corr {d['zero_lag_corr']:.3f} | avg-ref corr {d['avg_ref_corr']:.3f} | "
                      f"best lag {d['best_lag']} (peak corr {d['lag_peak_corr']:.3f}) | same-channel-is-best {d['same_channel_is_best']:.2f} | "
                      f"std ratio B/A {d['std_ratio']:.3g}")
    if allres:
        med = {k: float(np.median([r[k] for r in allres])) for k in allres[0]}
        print("\n=== medians over all examined pairs ===")
        print({k: round(v, 4) for k, v in med.items()})
        print("\nREADING:")
        for m in verdict(med):
            print("  -", m)
    else:
        print("No respiration-matched trial pairs found; run compare_epochs.py first.")


if __name__ == "__main__":
    sys.exit(main())
