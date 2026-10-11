"""scripts/finalize_release.py fills every placeholder (LICENSE, URL, DOIs, pyrfume commit, pins) and is idempotent."""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
FIX = os.path.join(os.path.dirname(__file__), "fixtures")
# fixture name -> relative path in the temporary release tree
FILES = {"README.template.md": "README.md", "CITATION.template.cff": "CITATION.cff", "DEPOSIT.template.md": "DEPOSIT.md",
         "environment.template.yml": "environment.yml", "goodscents_source.template.json": "config/goodscents_selected_source.json"}


def make_tree(tmp_path):
    """A pristine (not yet finalized) release tree, built from the fixtures."""
    for src, dst in FILES.items():
        os.makedirs(os.path.dirname(tmp_path / dst), exist_ok=True)
        shutil.copy(os.path.join(FIX, src), tmp_path / dst)


def find(f):
    """The live file (CITATION.cff sits at the top level of a larger repository when this code lives in a subfolder; see scripts/place_in_monorepo.py)."""
    here = os.path.join(ROOT, f)
    return here if os.path.exists(here) else os.path.join(os.path.dirname(ROOT), f)


def run(tmp, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "finalize_release.py"), "--root", str(tmp), *args], capture_output=True, text=True)


def test_finalize_fills_everything(tmp_path):
    make_tree(tmp_path)
    (tmp_path / "versions.txt").write_text("python      3.13.11  (x)\nh5py        3.15.1\npywt        1.9.0\nyaml        6.0.3\nfire        0.7.1\nstatsmodels 0.14.5\nsnakemake   9.27.0\n")
    dry = run(tmp_path, "--license", "MIT", "--holder", "A B", "--repo-url", "https://github.com/u/r")
    assert dry.returncode == 0 and "DRY RUN" in dry.stdout and not (tmp_path / "LICENSE").exists()
    args = ["--license", "MIT", "--holder", "A B and C D", "--repo-url", "https://github.com/u/r", "--paper-doi", "10.1073/pnas.1",
            "--data-doi", "10.5281/zenodo.1", "--pyrfume-commit", "abc123", "--versions", str(tmp_path / "versions.txt"), "--apply"]
    r = run(tmp_path, *args)
    assert r.returncode == 0, r.stderr
    assert "No TODO placeholders left" in r.stdout, r.stdout
    assert "MIT License" in (tmp_path / "LICENSE").read_text() and "A B and C D" in (tmp_path / "LICENSE").read_text()
    readme = (tmp_path / "README.md").read_text()
    assert "git clone https://github.com/u/r.git && cd r" in readme and "10.5281/zenodo.1" in readme and "10.1073/pnas.1" in readme
    assert 'repository-code: "https://github.com/u/r"' in (tmp_path / "CITATION.cff").read_text() and "license: MIT" in (tmp_path / "CITATION.cff").read_text()
    assert json.loads((tmp_path / "config/goodscents_selected_source.json").read_text())["commit"] == "abc123"
    env = (tmp_path / "environment.yml").read_text()
    for pin in ("h5py==3.15.1", "pywavelets==1.9.0", "pyyaml==6.0.3", "snakemake==9.27.0", "statsmodels==0.14.5", "fire==0.7.1"):
        assert pin in env, pin
    again = run(tmp_path, *args)                                           # idempotent: a second run changes nothing
    assert again.returncode == 0 and "nothing to change" in again.stdout.replace("APPLIED: ", ""), again.stdout


def test_forthcoming_paper_doi(tmp_path):
    make_tree(tmp_path)
    r = run(tmp_path, "--license", "BSD-3-Clause", "--holder", "X", "--repo-url", "https://github.com/u/r", "--paper-doi", "forthcoming",
            "--data-doi", "10.5281/zenodo.1", "--apply")
    assert r.returncode == 0 and "No TODO placeholders left" in r.stdout, r.stdout
    assert "DOI forthcoming" in (tmp_path / "README.md").read_text()


def test_cc_by_nc_license_text_is_written(tmp_path):
    make_tree(tmp_path)
    r = run(tmp_path, "--license", "CC-BY-NC-4.0", "--holder", "Alisa Y. Hathaway", "--data-license", "CC-BY-NC-4.0", "--repo-url", "https://github.com/u/r",
            "--paper-doi", "forthcoming", "--data-doi", "10.5281/zenodo.1", "--apply")
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "LICENSE").read_text()
    assert "Copyright (c)" in text and "Alisa Y. Hathaway" in text and "Attribution-NonCommercial 4.0 International" in text and len(text) > 15000
    assert "license: CC-BY-NC-4.0" in (tmp_path / "CITATION.cff").read_text()
    assert "CC-BY-NC-4.0 (code), CC-BY-NC-4.0 (data deposit)" in (tmp_path / "README.md").read_text()


def test_placeholders_match_the_script_patterns():
    """Before release only: the live README / CITATION.cff still contain the placeholder sentences finalize_release.py replaces.  Skipped once they are finalized."""
    readme = open(find("README.md"), encoding="utf-8").read()
    if "https://github.com/TODO/TODO.git" not in readme:                     # finalize_release.py has already been applied (fully or partly)
        import pytest
        pytest.skip("README.md is already finalized")
    for pattern in ("PNAS (2026). DOI: `TODO`", "(DOI: `TODO`; contents", "git clone https://github.com/TODO/TODO.git && cd TODO",
                    "Please cite the paper (DOI `TODO`)", "and the data deposit (DOI `TODO`)", "License: `TODO`."):
        assert pattern in readme, pattern
    cff = open(find("CITATION.cff"), encoding="utf-8").read()
    assert 'repository-code: "https://github.com/TODO/TODO"' in cff and "# license: TODO" in cff
