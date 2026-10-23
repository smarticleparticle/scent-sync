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


def test_filter_subjects_removes_only_other_participants():
    obj = {"EEG": {"water": {"22": [1], "19": [2], "30": [3]}}, "meta": {"19": {"x": 1}, "22": {"x": 2}}, "counts": {"1": 5, "2": 6},
           "subjects": ["22", "19"], "lookup": {"19": "a"}}
    out, removed = deposit.filter_subjects(obj, keep={"22"})
    assert set(out["EEG"]["water"]) == {"22"} and set(out["meta"]) == {"22"}
    assert removed == {"19": 2, "30": 1}
    assert out["counts"] == {"1": 5, "2": 6}                 # a dict without any kept participant is not a subject dict: untouched
    assert out["lookup"] == {"19": "a"}
    assert out["subjects"] == ["22", "19"]                   # lists of IDs are not rewritten (the audit looks for them)


def test_make_deposit_analyzed_only_leaves_out_the_other_participants(tmp_path):
    import subprocess
    import yaml
    src = tmp_path / "src"
    (src / "info").mkdir(parents=True)
    (src / "ersp").mkdir()
    def dump(name, obj):
        np.save(src / name, obj, allow_pickle=True)
    dump("epochs.npy", {"EEG": {"water": {"22": np.zeros(3), "19": np.ones(3)}}, "respiration": {"water": {"22": np.zeros(3), "19": np.ones(3)}}})
    dump("t1.npy", {"timestamps": {"water": {"22": np.arange(3.0), "30": np.arange(3.0)}}})
    dump("t2.npy", {"metadata": {"22": {"bad_channels": ["E1"]}, "25": {"bad_channels": []}}})
    dump("psd.npy", {"psd": {"22": {"E1": 1}, "19": {"E1": 2}}})
    dump("trial.npy", {"subject": "22", "trials": np.zeros((2, 3))})
    for f in ("phase_aligned_fullbreath_1deg.npy", "time_aligned_long.npy", "time_aligned_short.npy"):
        dump("ersp/" + f, {"water": {"22": {0: np.zeros(2)}}})
    for sid in ("22", "19", "20"):
        (src / "info" / f"subject{sid}_info.fif").write_bytes(b"i" + sid.encode())
    cfg = {"paths": {"epochs": str(src / "epochs.npy"), "info_dir": str(src / "info"), "epochs_v2_for_timing": str(src / "t1.npy"),
                     "continuous_respiration_v2": str(src / "t2.npy"), "psd_cache": str(src / "psd.npy"), "pac_results": str(src / "none.npy"),
                     "full_recordings_dir": str(src / "none"), "results": str(src)},
           "fig1_resp": {"trial_file": str(src / "trial.npy")}, "subjects": {"normosmia": ["22"]}}
    yaml.safe_dump(cfg, open(tmp_path / "cfg.yaml", "w"))
    out = tmp_path / "deposit"
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "make_deposit.py"), "--config", str(tmp_path / "cfg.yaml"), "--out-dir", str(out),
                        "--include-ersp", "--copy-all", "--analyzed-only"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "info" / "subject22_info.fif").exists() and not (out / "info" / "subject19_info.fif").exists() and not (out / "info" / "subject20_info.fif").exists()
    ep = np.load(out / "epochs.npy", allow_pickle=True).item()
    assert set(ep["EEG"]["water"]) == {"22"} and set(ep["respiration"]["water"]) == {"22"}
    assert set(np.load(out / "psd.npy", allow_pickle=True).item()["psd"]) == {"22"}
    assert set(np.load(out / "t1.npy", allow_pickle=True).item()["timestamps"]["water"]) == {"22"}
    assert "subject 19" in r.stdout and "subject30" not in r.stdout and "subject 30" in r.stdout and "subject 25" in r.stdout
    status = deposit.verify_manifest(str(out / "MANIFEST.sha256"), str(out))
    assert status and all(v == "ok" for v in status.values())
    mtimes = {round(os.path.getmtime(os.path.join(dp, f))) for dp, _, fs in os.walk(out) for f in fs}
    assert len(mtimes) == 1                                                  # every file carries the same (neutral) date
    assert os.path.exists(src / "epochs.npy") and set(np.load(src / "epochs.npy", allow_pickle=True).item()["EEG"]["water"]) == {"22", "19"}   # the originals are untouched
