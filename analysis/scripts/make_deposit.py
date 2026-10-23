#!/usr/bin/env python
"""Build the derived files of the data deposit and a checksum manifest.

What goes into the deposit (see DEPOSIT.md):
  * epochs_eeg_respiration_withtimestamps_v2.npy   preprocessed EEG epochs + respiration epochs   (large; used as is)
  * info/subject*_info.fif                         channel names / montage                        (18 KB each)
  * fig1_example_trial.npy                           one subject's humidity snippets for Fig. 1 (right panel)  (made by `snakemake fig1_trial`)
  * pac_results_slim.npy                           PAC results without per-surrogate arrays       (created here)
  * full_respiration_EEG_humidity_meta.npy         channel info / bad channels                    (created here)
  * psd_cache.npy                                  Welch spectra of all channels (FOOOF analysis) (made by `snakemake psd_cache`)
  * (optional) ERSP files                          saves hours of wavelet computation

NOT deposited (available on request): raw EEGLAB .mat, olfactometer CSVs, continuous EEG (several GB each), whole-session respiration (odor-arrival timing check).

    python scripts/make_deposit.py --config config/config.yaml --out-dir deposit [--include-ersp]
"""
import argparse
import os
import re
import shutil

import numpy as np
import yaml

from olfaction_eeg import compat, deposit, recordings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--out-dir", default="deposit")
    ap.add_argument("--include-ersp", action="store_true", help="also list the ERSP result files (~1 GB)")
    ap.add_argument("--ersp-dir", default=None, help="folder with the three ERSP files (default: <paths.results>/ersp)")
    ap.add_argument("--pac-results", default=None, help="the full PAC results file (default: paths.pac_results)")
    ap.add_argument("--recordings-dir", default=None, help="folder with full_respiration_EEG_humidity.npy (default: paths.full_recordings_dir)")
    ap.add_argument("--psd-cache", default=None, help="the spectra cache (default: paths.psd_cache)")
    ap.add_argument("--copy-all", action="store_true",
                    help="copy every listed file into --out-dir (otherwise large existing files are only referenced), so that the folder is complete")
    ap.add_argument("--analyzed-only", action="store_true",
                    help="only the participants of subjects.normosmia (the ones in the paper's analyses): other participants' entries are removed from every file, their montage files are not copied")
    ap.add_argument("--epochs", default=None, help="the epochs file the paper's ERSPs were computed from (default: paths.epochs)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    P = dict(cfg["paths"])
    for key, val in (("pac_results", a.pac_results), ("full_recordings_dir", a.recordings_dir), ("psd_cache", a.psd_cache), ("epochs", a.epochs)):
        if val:
            P[key] = val
    out = a.out_dir
    keep = set(cfg["subjects"]["normosmia"]) if a.analyzed_only else None
    os.makedirs(os.path.join(out, "info"), exist_ok=True)
    entries, skipped, dropped_info, filter_report = [], [], [], []

    def add(path, name):
        entries.append((path, name))

    # --- files used as they are
    if os.path.exists(P["epochs"]):
        add(P["epochs"], os.path.basename(P["epochs"]))
    else:
        skipped.append(P["epochs"])
    info_dir = P["info_dir"]
    for f in sorted(os.listdir(info_dir)) if os.path.isdir(info_dir) else []:
        if f.endswith(".fif"):
            m = re.search(r"subject(\d+)_info\.fif$", f)
            if keep is not None and m and m.group(1) not in keep:
                dropped_info.append(f)
                continue
            shutil.copy(os.path.join(info_dir, f), os.path.join(out, "info", f))
            add(os.path.join(out, "info", f), f"info/{f}")

    # --- derived files
    # (the whole-session respiration used for the odor-arrival timing check, Section 1.8 / Tables S5-S6, is NOT deposited: available on request)
    fig1_file = (cfg.get("fig1_resp") or {}).get("trial_file")                  # made by `snakemake fig1_trial` (needs the raw data)
    if fig1_file:
        if os.path.exists(fig1_file):
            add(fig1_file, os.path.basename(fig1_file))
        else:
            skipped.append(fig1_file)
    rec_dir = P["full_recordings_dir"]
    for kind in ("humidity",):
        src = os.path.join(rec_dir, f"full_respiration_EEG_{kind}.npy")
        if os.path.exists(src):
            full = compat.load_pickled_dict(src)
            # (the continuous respiration of this recording is NOT deposited: only the PAC computation reads it, and the PAC results themselves are deposited)
            if kind == "humidity":
                dst = os.path.join(out, "full_respiration_EEG_humidity_meta.npy")
                np.save(dst, recordings.make_recordings_meta(full), allow_pickle=True)
                add(dst, os.path.basename(dst))
            del full
        else:
            skipped.append(src)
    if os.path.exists(P["pac_results"]):
        dst = os.path.join(out, "pac_results_slim.npy")
        np.save(dst, deposit.slim_pac_results(np.load(P["pac_results"], allow_pickle=True).item()), allow_pickle=True)
        add(dst, os.path.basename(dst))
    else:
        skipped.append(P["pac_results"])
    if os.path.exists(P["psd_cache"]):
        add(P["psd_cache"], os.path.basename(P["psd_cache"]))
    else:
        skipped.append(P["psd_cache"])
    if a.include_ersp:
        for f in ("phase_aligned_fullbreath_1deg.npy", "time_aligned_long.npy", "time_aligned_short.npy"):
            p = os.path.join(a.ersp_dir or os.path.join(P["results"], "ersp"), f)
            if os.path.exists(p):
                add(p, f"ersp/{f}")
            else:
                skipped.append(p)

    if keep is not None:                                                                  # remove everybody who is not in the analyses from every data file
        for i, (path, name) in enumerate(entries):
            if not name.endswith(".npy"):
                continue
            obj = compat.load_pickled_dict(path)
            obj, removed = deposit.filter_subjects(obj, keep)
            filter_report.append((name, removed))
            if removed:
                dst = os.path.join(out, name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                np.save(dst, obj, allow_pickle=True)
                entries[i] = (dst, name)
            del obj
    if a.copy_all:
        for i, (path, name) in enumerate(entries):
            dst = os.path.join(out, name)
            if os.path.abspath(path) != os.path.abspath(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(path) and (
                        os.path.getsize(path) > 50e6 or deposit.sha256(dst) == deposit.sha256(path)):
                    print(f"up to date: {name}", flush=True)          # identical copy from an earlier run (large files: same size)
                else:
                    print(f"copying {name} ...", flush=True)
                    shutil.copy2(path, dst)
                entries[i] = (dst, name)
    manifest = os.path.join(out, "MANIFEST.sha256")
    deposit.write_manifest(entries, manifest)
    total = 0
    print(f"{'file':55s} {'size':>12s}")
    for path, name in entries:
        sz = os.path.getsize(path)
        total += sz
        print(f"{name:55s} {sz / 1e6:10.1f} MB")
    print(f"{'TOTAL':55s} {total / 1e6:10.1f} MB\nManifest: {manifest}")
    if skipped:
        print("\nNot found (skipped):\n  " + "\n  ".join(skipped))
    if keep is not None:
        print(f"\nOnly the {len(keep)} analysed participants are included. Left out:")
        print("  montage files: " + (", ".join(dropped_info) if dropped_info else "none"))
        for name, removed in filter_report:
            print(f"  {name}: " + (", ".join(f"subject {k} ({n}x)" for k, n in sorted(removed.items())) if removed else "no other participants found"))
    deposit.normalise_mtimes(out)                                                        # no file carries the dates it was made on
    print("\nFile modification times in the deposit folder were set to 2026-01-01.")


if __name__ == "__main__":
    main()
