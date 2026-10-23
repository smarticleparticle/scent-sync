#!/usr/bin/env python
"""Print (and optionally save) the versions of everything the analysis depends on.

Run it inside the SAME environment/kernel you use for the analysis:
    python scripts/print_versions.py                      # print
    python scripts/print_versions.py results/versions.txt # print + save
From inside a notebook:  %run scripts/print_versions.py
"""
import importlib
import platform
import sys

PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "mne", "pywt", "statsmodels", "matplotlib",
            "seaborn", "h5py", "fooof", "pingouin", "fire", "yaml", "snakemake"]


def collect():
    lines = [f"python      {sys.version.split()[0]}  ({platform.platform()})", f"executable  {sys.executable}"]
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            lines.append(f"{name:<11} {getattr(m, '__version__', '(no __version__)')}")
        except Exception:
            lines.append(f"{name:<11} not installed")
    return lines


if __name__ == "__main__":
    text = "\n".join(collect())
    print(text)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(text + "\n")
