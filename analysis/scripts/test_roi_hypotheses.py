#!/usr/bin/env python
"""Which ROI channel set reproduces the paper's phase-aligned ERSP?

generate_ersp.py averages the wavelet power over the ROI channels that are shared by ALL info files it finds
(`shared_roi_channels`).  If a different set of info files (or channels) was used when the paper's ERSP file was made, the
averaged power differs slightly.  This script recomputes ONE subject x condition (first few trials) under several candidate
channel sets and compares each with the paper's arrays.  A candidate that reproduces them to ~1e-9 is the answer.

    python scripts/test_roi_hypotheses.py --epochs data/preprocessed/epochs_eeg_respiration.npy \\
        --info-dir data/preprocessed/info --paper-ersp PAPER_phase_aligned.npy --cond citrus --subject 22

Takes about a minute per candidate (4 trials x ~54 channels).  Also lists which info file lacks which ROI channel.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_ersp as g  # noqa: E402


def read_channel_names(info_dir):
    import mne
    out = {}
    for f in sorted(glob.glob(os.path.join(info_dir, "subject*_info.fif"))):
        m = re.search(r"subject(\d+)_info\.fif$", f)
        if m:
            out[m.group(1)] = list(mne.io.read_info(f, verbose=False)["ch_names"])
    return out


def shared_roi(names_by_subject, subjects=None):
    sets = [set(v) for k, v in names_by_subject.items() if subjects is None or k in subjects]
    return sorted(set(g.roi_channels).intersection(*sets), key=lambda c: int(c[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--info-dir", required=True)
    ap.add_argument("--paper-ersp", required=True)
    ap.add_argument("--cond", default="citrus")
    ap.add_argument("--subject", default="22")
    ap.add_argument("--n-trials", type=int, default=4, help="first trials to recompute (the arrays are per trial)")
    ap.add_argument("--analysed-subjects", nargs="*", default=None, help="default: subjects.normosmia of config/config.yaml")
    ap.add_argument("--exclude", default="30")
    a = ap.parse_args()

    names = read_channel_names(a.info_dir)
    print(f"info files: {sorted(names, key=int)}")
    print("\nROI channels (of %d in the script's list) missing from each info file:" % len(g.roi_channels))
    for s in sorted(names, key=int):
        miss = [c for c in g.roi_channels if c not in names[s]]
        print(f"  subject {s}: {len(names[s])} channels; missing {miss if miss else '-'}")

    epochs = np.load(a.epochs, allow_pickle=True).item()
    paper = np.load(a.paper_ersp, allow_pickle=True).item()
    in_epochs = sorted(set().union(*[set(epochs["EEG"][c]) for c in epochs["EEG"]]), key=int)
    if a.analysed_subjects is None:
        cfg = yaml.safe_load(open(os.path.join(HERE, "..", "config", "config.yaml")))
        a.analysed_subjects = cfg["subjects"]["normosmia"]
    cand = {
        "A  all script ROI channels present in this subject's own info": [c for c in g.roi_channels if c in names[a.subject]],
        "B  shared across ALL info files in the folder (what the pipeline run used)": shared_roi(names),
        "C  shared across the info files of subjects present in the epochs file": shared_roi(names, set(in_epochs)),
        "D  ... C without the excluded subject %s" % a.exclude: shared_roi(names, set(in_epochs) - {a.exclude}),
        "E  shared across the %d analysed subjects" % len(a.analysed_subjects): shared_roi(names, set(a.analysed_subjects)),
    }
    uniq = {}
    for label, chans in cand.items():
        uniq.setdefault(tuple(chans), []).append(label)

    eeg = epochs["EEG"][a.cond][a.subject][: a.n_trials]
    resp = epochs["respiration"][a.cond][a.subject][: a.n_trials]
    ref = paper[a.cond][a.subject]
    ch_list = names[a.subject]
    print(f"\nrecomputing {a.cond} / subject {a.subject}, first {a.n_trials} trials, for {len(uniq)} distinct channel sets ...")
    best = None
    for chans, labels in uniq.items():
        roi_idx = [ch_list.index(c) for c in chans]
        counts = {}
        res = g.compute_subject_condition_phase_aligned(eeg, resp, a.subject, a.cond, counts, roi_idx, shared_roi_channels=list(chans))
        keys = sorted(set(res or {}) & set(ref))
        if not keys:
            print(f"\n[{len(chans)} channels] no comparable trials")
            continue
        d = np.concatenate([np.abs(res[k] - ref[k])[~np.isnan(res[k]) & ~np.isnan(ref[k])] for k in keys])
        rms, mx = float(np.sqrt(np.mean(d ** 2))), float(d.max())
        print(f"\n[{len(chans)} channels]  matches candidates: {' | '.join(l.split('  ')[0] for l in labels)}")
        print(f"    trials compared: {keys}   max |diff| {mx:.3g}   rms diff {rms:.3g}   {'<-- EXACT MATCH' if mx < 1e-9 else ''}")
        if best is None or rms < best[0]:
            best = (rms, mx, chans, labels)
    if best:
        print("\n=== RESULT ===")
        if best[1] < 1e-9:
            print(f"The paper's arrays are reproduced exactly with {len(best[2])} channels (candidates {', '.join(l.split('  ')[0] for l in best[3])}).")
            print("Channels:", " ".join(best[2]))
            print("-> use the info files of that candidate (or set the ROI explicitly).")
        else:
            print(f"No candidate reproduces the paper exactly (best rms {best[0]:.3g} with {len(best[2])} channels).")
            print("-> the difference is not just the channel set: script version, library versions or other processing.")


if __name__ == "__main__":
    main()
