#!/usr/bin/env python
"""Check a downloaded copy of the data deposit against its checksum manifest, and say which outputs it can regenerate.

    python scripts/verify_data.py --manifest MANIFEST.sha256 --root data/preprocessed
"""
import argparse
import sys

from olfaction_eeg import deposit

NEEDED_FOR = {
    "epochs_eeg_respiration_withtimestamps_v2.npy": "ERSPs -> Table S1/S4, Fig 3-5, S3, S4, Section 1.7",
    "pac_results_slim.npy": "Fig 2, Section 1.1-1.2, Table S2",
    "psd_cache.npy": "Section 1.2, Table S2 (FOOOF analysis)",
    "full_respiration_EEG_humidity_meta.npy": "Fig 2 (channel positions)",
    "continuous_respiration_v2.npy": "odor-arrival timing (Section 1.8; Tables S5, S6)",
    "epochs_v2_for_timing.npy": "odor-arrival timing (Section 1.8; Tables S5, S6)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    status = deposit.verify_manifest(a.manifest, a.root)
    bad = {n: s for n, s in status.items() if s != "ok"}
    for n, s in status.items():
        note = f"   (needed for: {NEEDED_FOR[n]})" if n in NEEDED_FOR and s != "ok" else ""
        print(f"[{s.upper():8s}] {n}{note}")
    print(f"\n{len(status) - len(bad)}/{len(status)} files verified")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
