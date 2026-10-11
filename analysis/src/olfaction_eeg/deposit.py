"""Helpers for building and checking the public data deposit (Zenodo).

The paper's raw EEG is not deposited; the deposit contains the *minimum derived data needed to regenerate every
figure and table*.  Large intermediate arrays that no downstream analysis reads are dropped, and every file is listed
with a SHA-256 checksum so that a downloaded copy can be verified (scripts/verify_data.py).
"""
import hashlib
import os

import numpy as np

# Per channel x band PAC result fields that the statistics / Fig. 2 / oscillatory-gating analyses read.  The two dropped
# arrays (1000 x 18 surrogate amplitude profiles and 1000 surrogate preferred phases) are ~95% of the file size and are not
# used by any analysis after the PAC computation itself.
PAC_FIELDS_KEPT = ("MI_observed", "MI_null_mean", "MI_null_std", "MI_null_distribution", "MI_null_ci_95", "p_value",
                   "z_score", "effect_size", "significant_p05", "significant_p01", "mean_amp_per_phase_bin",
                   "preferred_phase_observed", "n_samples", "n_valid_phase_samples", "duration_sec", "phase_method")


def slim_pac_results(pac_results, fields=PAC_FIELDS_KEPT):
    """Copy of a PAC results dict without the large per-surrogate arrays (metadata is kept in full)."""
    slim = {"metadata": pac_results["metadata"], "subject_level": {}}
    for subj, channels in pac_results["subject_level"].items():
        slim["subject_level"][subj] = {}
        for ch, bands in channels.items():
            slim["subject_level"][subj][ch] = {
                band: (None if res is None else {k: res[k] for k in fields if k in res}) for band, res in bands.items()}
    return slim


def extract_continuous_respiration(full_recordings):
    """Continuous respiration, event times and metadata of a full_respiration_EEG_*.npy dict -- WITHOUT the continuous EEG.

    This is all the breath-timing analyses (odor arrival, breath durations, retention vs breathing rate) need, and it is
    megabytes instead of gigabytes.
    """
    return {k: full_recordings[k] for k in ("respiration", "event_times", "metadata") if k in full_recordings}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(entries, out_path):
    """entries: [(path_on_disk, name_in_deposit)] -> text manifest 'sha256  size_bytes  name' (one line per file)."""
    lines = []
    for path, name in entries:
        lines.append(f"{sha256(path)}  {os.path.getsize(path)}  {name}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def read_manifest(path):
    out = {}
    for line in open(path):
        if line.strip():
            digest, size, name = line.rstrip("\n").split(None, 2)
            out[name] = (digest, int(size))
    return out


def verify_manifest(manifest, root):
    """Compare files below ``root`` with a manifest.  Returns {name: 'ok' | 'missing' | 'size' | 'checksum'}."""
    status = {}
    for name, (digest, size) in read_manifest(manifest).items():
        p = os.path.join(root, name)
        if not os.path.exists(p):
            status[name] = "missing"
        elif os.path.getsize(p) != size:
            status[name] = "size"
        else:
            status[name] = "ok" if sha256(p) == digest else "checksum"
    return status


def extract_v2_for_timing(epochs_v2):
    """The part of the v2 epochs file the timing analysis needs: respiration epochs, timestamps and the EEG array SHAPES (no EEG)."""
    return {"respiration": epochs_v2["respiration"], "timestamps": epochs_v2["timestamps"],
            "eeg_shapes": {c: {s: tuple(a.shape) for s, a in subj.items()} for c, subj in epochs_v2["EEG"].items()}}
