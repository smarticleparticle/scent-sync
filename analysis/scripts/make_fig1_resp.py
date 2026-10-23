#!/usr/bin/env python
"""Fig. 1 (right panel): example respiration trace around the TTL pulse.  Replaces analysis-notebook cell 115.

Two steps (the first needs the raw data, which is not deposited; the second needs only the tiny file the first one writes):

  extract:  raw EEGLAB .mat + olfactometer CSV  ->  fig1_example_trial.npy   (one subject's QC-passed humidity snippets)
      python scripts/make_fig1_resp.py extract --subject 22 --csv data/raw/olfactometer/OlfactometerSubject22.csv \
          --mat data/raw/eeglab/22/22_EEG_preprocessed.mat --params results/fig1/params.yaml --out data/preprocessed/fig1_example_trial.npy
  plot:     fig1_example_trial.npy  ->  figures/fig1_respiration_example.{pdf,png}
      python scripts/make_fig1_resp.py plot --trial-file data/preprocessed/fig1_example_trial.npy --out-prefix figures/fig1_respiration_example
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import yaml

from olfaction_eeg import fig1_resp


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--subject", required=True)
    e.add_argument("--csv", required=True)
    e.add_argument("--mat", required=True)
    e.add_argument("--params", required=True)
    e.add_argument("--out", required=True)
    pl = sub.add_parser("plot")
    pl.add_argument("--trial-file", required=True)
    pl.add_argument("--out-prefix", required=True)
    pl.add_argument("--trial-idx", type=int, default=None)
    a = ap.parse_args()

    if a.cmd == "extract":
        p = yaml.safe_load(open(a.params))
        offsets = {str(k): int(v) for k, v in p["offsets"].items()}
        trials = fig1_resp.extract_respiratory_data(a.subject, a.csv, a.mat, offsets)
        if trials is None or len(trials) == 0:
            raise SystemExit(f"No accepted respiration trials found for subject {a.subject}.")
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.save(a.out, {"subject": a.subject, "trials": trials, "resp_sfreq": 8.75714285714, "pre_ttl": 4, "post_ttl": 4})
        print(f"{len(trials)} accepted trials for subject {a.subject}; wrote {a.out} ({os.path.getsize(a.out) / 1e3:.0f} kB)")
    else:
        d = np.load(a.trial_file, allow_pickle=True).item()
        idx = 0 if a.trial_idx is None else a.trial_idx
        os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
        fig = fig1_resp.plot_example_trial(d["trials"][idx], trial_idx=idx, pre_ttl=d["pre_ttl"], post_ttl=d["post_ttl"])
        fig.savefig(a.out_prefix + ".pdf", bbox_inches="tight")
        print(f"wrote {a.out_prefix}.pdf (subject {d['subject']}, trial {idx + 1} of {len(d['trials'])})")


if __name__ == "__main__":
    main()
