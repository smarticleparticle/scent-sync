#!/usr/bin/env python
"""Does an OLDER version of generate_ersp.py reproduce the paper's phase-aligned ERSP arrays?

Loads each given script as a module, recomputes ONE subject x condition (first few trials) with that script's own ROI list
and processing parameters, and compares with the paper's arrays.  An exact match (difference < 1e-9) identifies the version.

    python scripts/test_script_version.py --epochs EPOCHS.npy --info-dir INFO_DIR --paper-ersp PAPER.npy \\
        --scripts path/to/ersp_versions/*.py [--cond citrus --subject 22 --n-trials 4]
"""
import argparse
import glob
import importlib.util
import inspect
import os
import re
import sys
import traceback

import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def find_roi(mod):
    for k in ("roi_channels", "ROI_CHANNELS"):
        if hasattr(mod, k):
            return list(getattr(mod, k))
    for k, v in vars(mod).items():
        if isinstance(v, list) and v and all(isinstance(x, str) and re.fullmatch(r"E\d+", x) for x in v):
            return list(v)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--info-dir", required=True)
    ap.add_argument("--paper-ersp", required=True)
    ap.add_argument("--scripts", nargs="+", required=True)
    ap.add_argument("--cond", default="citrus")
    ap.add_argument("--subject", default="22")
    ap.add_argument("--n-trials", type=int, default=4)
    a = ap.parse_args()

    import mne
    names = {}
    for f in sorted(glob.glob(os.path.join(a.info_dir, "subject*_info.fif"))):
        m = re.search(r"subject(\d+)_info\.fif$", f)
        if m:
            names[m.group(1)] = list(mne.io.read_info(f, verbose=False)["ch_names"])
    epochs = np.load(a.epochs, allow_pickle=True).item()
    paper = np.load(a.paper_ersp, allow_pickle=True).item()
    eeg = epochs["EEG"][a.cond][a.subject][: a.n_trials]
    resp = epochs["respiration"][a.cond][a.subject][: a.n_trials]
    ref = paper[a.cond][a.subject]

    paths = []
    for p in a.scripts:
        paths += sorted(glob.glob(os.path.expanduser(p)))
    print(f"testing {len(paths)} script version(s) on {a.cond} / subject {a.subject}, first {a.n_trials} trials\n")
    summary = []
    for i, path in enumerate(paths):
        label = os.path.basename(path)
        try:
            mod = load_module(path, f"ersp_version_{i}")
            roi = find_roi(mod)
            fn = getattr(mod, "compute_subject_condition_phase_aligned", None)
            if roi is None or fn is None:
                raise RuntimeError(f"no ROI list ({roi is None}) or no compute_subject_condition_phase_aligned ({fn is None}); "
                                   f"functions found: {[k for k, v in vars(mod).items() if inspect.isfunction(v)][:12]}")
            shared = sorted(set(roi).intersection(*[set(v) for v in names.values()]), key=lambda c: int(c[1:]))
            roi_idx = [names[a.subject].index(c) for c in shared]
            cand = dict(eeg_data=eeg, resp_data=resp, subj_id=a.subject, cond_name=a.cond, trial_counts_global={},
                        roi_indices=roi_idx, shared_roi_channels=shared, save_channels=False)
            kw = {k: v for k, v in cand.items() if k in inspect.signature(fn).parameters}
            res = fn(**kw)
            keys = sorted(set(res or {}) & set(ref))
            d = np.concatenate([np.abs(res[k] - ref[k])[~np.isnan(res[k]) & ~np.isnan(ref[k])] for k in keys])
            rms, mx = float(np.sqrt(np.mean(d ** 2))), float(d.max())
            status = "EXACT MATCH" if mx < 1e-9 else ""
            print(f"[{label}] ROI {len(roi)} listed / {len(shared)} usable; trials {keys}; max |diff| {mx:.3g}  rms {rms:.3g}  {status}", flush=True)
            summary.append((rms, mx, label, len(shared)))
        except Exception as e:
            print(f"[{label}] could not be tested: {type(e).__name__}: {str(e)[:300]}", flush=True)
            traceback.print_exc(limit=1)
    print("\n=== RESULT ===")
    if summary:
        summary.sort()
        exact = [s for s in summary if s[1] < 1e-9]
        if exact:
            print("These version(s) reproduce the paper's arrays exactly:", ", ".join(s[2] for s in exact))
        else:
            print(f"No version reproduces the paper exactly. Closest: {summary[0][2]} (rms {summary[0][0]:.3g}).")
            print("-> the difference is probably in library versions (PyWavelets, SciPy, NumPy) or in a version of the script that is not in git.")


if __name__ == "__main__":
    main()
