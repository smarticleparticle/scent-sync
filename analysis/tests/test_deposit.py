import os
import sys

import numpy as np
import pytest

mne = pytest.importorskip("mne")
sys.path.insert(0, os.path.dirname(__file__))
import make_synthetic_pac as msp  # noqa: E402

from olfaction_eeg import deposit, pac_stats  # noqa: E402


def fat_pac(subjects, info, n_surr=50):
    """Synthetic PAC results carrying the big per-surrogate arrays of the real file."""
    res = msp.make_pac_results(subjects, info, n_surrogates=n_surr, seed=1)
    for s in res["subject_level"].values():
        for ch in s.values():
            for band, r in ch.items():
                if r is not None:
                    r["surrogate_mean_amp_bins"] = np.random.rand(n_surr, 18)
                    r["surrogate_preferred_phases"] = np.random.rand(n_surr)
                    r["mean_amp_per_phase_bin"] = np.random.rand(18)
    return res


def test_slim_pac_keeps_statistics_identical_and_is_much_smaller(tmp_path):
    info = msp.make_info(n_ch=25, seed=2)
    subjects = [str(s) for s in range(20, 32)]
    full = fat_pac(subjects, info, n_surr=200)
    slim = deposit.slim_pac_results(full)
    a, _ = pac_stats.compute_subject_wholehead_stats_with_ids(full, min_channels=20)
    b, _ = pac_stats.compute_subject_wholehead_stats_with_ids(slim, min_channels=20)
    assert a.equals(b)
    assert pac_stats.compute_group_stats(a).equals(pac_stats.compute_group_stats(b))
    np.save(tmp_path / "full.npy", full, allow_pickle=True)
    np.save(tmp_path / "slim.npy", slim, allow_pickle=True)
    assert os.path.getsize(tmp_path / "slim.npy") < 0.5 * os.path.getsize(tmp_path / "full.npy")
    r = slim["subject_level"][subjects[0]]["E1"]["alpha"]
    assert "surrogate_mean_amp_bins" not in r and "MI_null_distribution" in r and "z_score" in r


def test_continuous_respiration_has_no_eeg():
    recs = msp.make_recordings(subjects=["22"], n_ch=2, duration_s=20, seed=1)
    recs["event_times"] = {"22": {"water": np.array([1.0])}}
    out = deposit.extract_continuous_respiration(recs)
    assert set(out) == {"respiration", "event_times", "metadata"} and "EEG" not in out


def test_manifest_roundtrip_detects_problems(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"hello")
    (tmp_path / "b.bin").write_bytes(b"world")
    deposit.write_manifest([(str(tmp_path / "a.bin"), "a.bin"), (str(tmp_path / "b.bin"), "b.bin"),], tmp_path / "M.sha256")
    assert set(deposit.verify_manifest(tmp_path / "M.sha256", tmp_path).values()) == {"ok"}
    (tmp_path / "b.bin").write_bytes(b"WORLD")                       # same size, different content
    (tmp_path / "a.bin").unlink()
    st = deposit.verify_manifest(tmp_path / "M.sha256", tmp_path)
    assert st == {"a.bin": "missing", "b.bin": "checksum"}


def test_make_deposit_copy_all_builds_a_verifiable_folder(tmp_path):
    """The deposit script with --copy-all: every listed file lands in the output folder and the manifest verifies against it."""
    import subprocess
    import yaml
    src = tmp_path / "src"
    (src / "info").mkdir(parents=True)
    (src / "ersp").mkdir()
    for name, content in (("epochs.npy", b"epochs"), ("t1.npy", b"t1"), ("t2.npy", b"t2"), ("psd.npy", b"psd"), ("trial.npy", b"trial"),
                          ("ersp/phase_aligned_fullbreath_1deg.npy", b"p"), ("ersp/time_aligned_long.npy", b"l"),
                          ("ersp/time_aligned_short.npy", b"s"), ("info/subject22_info.fif", b"i22")):
        (src / name).write_bytes(content)
    cfg = {"paths": {"epochs": str(src / "epochs.npy"), "info_dir": str(src / "info"), "epochs_v2_for_timing": str(src / "t1.npy"),
                     "continuous_respiration_v2": str(src / "t2.npy"), "psd_cache": str(src / "psd.npy"), "pac_results": str(src / "none.npy"),
                     "full_recordings_dir": str(src / "none"), "results": str(src)},
           "fig1_resp": {"trial_file": str(src / "trial.npy")}}
    yaml.safe_dump(cfg, open(tmp_path / "cfg.yaml", "w"))
    root = os.path.dirname(os.path.dirname(__file__))
    out = tmp_path / "deposit"
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "make_deposit.py"), "--config", str(tmp_path / "cfg.yaml"),
                        "--out-dir", str(out), "--include-ersp", "--copy-all"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for name in ("epochs.npy", "t1.npy", "t2.npy", "psd.npy", "trial.npy", "info/subject22_info.fif", "ersp/time_aligned_long.npy", "MANIFEST.sha256"):
        assert (out / name).exists(), name
    status = deposit.verify_manifest(str(out / "MANIFEST.sha256"), str(out))
    assert status and all(v == "ok" for v in status.values())
    assert "none.npy" in r.stdout                                     # the files that were not found are reported, not silently dropped
