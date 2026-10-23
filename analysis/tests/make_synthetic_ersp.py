"""Generate SYNTHETIC ERSP result files directly (no wavelet step), for fast statistics / figure tests.

Structure matches what scripts/generate_ersp.py writes:
    {condition: {subject: {trial_index: array(n_freqs, n_x)}}, "freqs": ..., "x": ...}

The synthetic effect (odor vs water) lives in theta (inhale half) and alpha (exhale half) of the
phase-locked data, is weak in the time-locked data, differs between odors, varies by subject and
DECAYS over trials (so habituation analyses have something to find).  Edge cases included: uneven trial
counts, a subject lacking one odor, a subject with too few trials for the late block.

Usage (as a module):  make(out_dir, subjects=..., seed=0) -> dict of file paths
"""
import os

import numpy as np

CONDITIONS = ["water", "peppermint", "oregano", "citrus"]
ODOR_STRENGTH = {"peppermint": 1.0, "oregano": 0.6, "citrus": 0.35}
NORMOSMIA = ["22", "23", "24", "26", "27", "28", "29", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"]


def _grid(kind, n_freqs=24):
    freqs = np.geomspace(2, 40, n_freqs)
    if kind == "phase":
        x = np.arange(360) + 0.5
    elif kind == "long":
        x = np.linspace(-0.5, 3.776, 108)
    elif kind == "short":
        x = np.linspace(-0.5, 1.996, 63)
    else:
        raise ValueError(kind)
    return freqs, x


def _effect(kind, freqs, x):
    f = freqs[:, None]
    theta, alpha = (f >= 4) & (f < 8), (f >= 8) & (f < 12)
    if kind == "phase":
        return 0.04 * (theta & (x[None, :] < 180)) - 0.04 * (alpha & (x[None, :] >= 180))
    return 0.02 * (theta & ((x[None, :] > 0) & (x[None, :] < 2.0))) - 0.01 * (alpha & (x[None, :] > 2.0))


def make_one(kind, subjects, rng, n_trials_range=(8, 11), missing=(("28", "peppermint"),), short_subject="40"):
    freqs, x = _grid(kind)
    eff = _effect(kind, freqs, x)
    out = {c: {} for c in CONDITIONS}
    for si, s in enumerate(subjects):
        amp = rng.uniform(0.4, 1.4)
        for c in CONDITIONS:
            if (s, c) in missing:
                continue
            n_tr = 6 if s == short_subject else int(rng.integers(*n_trials_range))
            trials = {}
            for t in range(n_tr):
                arr = rng.normal(scale=1.0, size=(len(freqs), len(x)))
                if c != "water":
                    arr = arr + ODOR_STRENGTH[c] * amp * np.exp(-t / 3.0) * eff
                trials[t] = arr
            out[c][s] = trials
    out["freqs"], out["x"] = freqs, x
    return out


def make(out_dir, subjects=None, seed=0):
    subjects = subjects or NORMOSMIA
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for kind, name in [("phase", "phase_aligned.npy"), ("long", "time_aligned_long.npy"), ("short", "time_aligned_short.npy")]:
        paths[kind] = os.path.join(out_dir, name)
        np.save(paths[kind], make_one(kind, subjects, rng))
    return paths


if __name__ == "__main__":
    import sys
    print(make(sys.argv[1] if len(sys.argv) > 1 else "tests/data/ersp"))
