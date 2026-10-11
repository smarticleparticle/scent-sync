"""Free-response identification score (word_scoring): offline checks with a deterministic stand-in for the embedding model."""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from olfaction_eeg import word_scoring as ws

ROOT = os.path.dirname(os.path.dirname(__file__))


class StubModel:
    """Deterministic fake sentence encoder (character-trigram hashing): similar words get similar vectors."""
    dim = 48

    def encode(self, text, convert_to_numpy=True, **kw):
        t = str(text).lower()
        v = np.zeros(self.dim)
        for g in [t[i:i + 3] for i in range(max(1, len(t) - 2))] + [t]:
            v += np.random.default_rng(int(hashlib.md5(g.encode()).hexdigest()[:8], 16)).normal(size=self.dim)
        return v / (np.linalg.norm(v) + 1e-12)


def profiles():
    found = ws.found_from_selected(pd.read_csv(os.path.join(ROOT, "config", "goodscents_selected.csv")))
    emp = ws.build_empirical_scoring_from_data({"goodscents": True}, found)
    return found, emp, ws.expand_profiles_with_synonyms(emp)


def test_committed_goodscents_subset_reproduces_the_published_profiles():
    found, emp, expd = profiles()
    assert [len(found[o]) for o in ["peppermint", "grapefruit", "oregano"]] == [8, 3, 4]
    assert [len(emp[o]) for o in ["peppermint", "grapefruit", "oregano"]] == [20, 9, 13]
    assert emp["peppermint"]["minty"] == pytest.approx(0.625) and emp["peppermint"]["mentholic"] == pytest.approx(0.5)
    assert emp["oregano"]["herbal"] == pytest.approx(1.0) and emp["grapefruit"]["citrus"] == pytest.approx(2 / 3)
    assert expd["peppermint"]["mint"] == pytest.approx(0.625)                 # synonym expansion: minty -> mint


def test_scoring_rule_branches():
    _, _, expd = profiles()
    model = StubModel()
    comb = ws.build_combined_profiles_50_50_v2(expd, model)
    assert ws.score_50_50_v2("peppermint", "peppermint", comb, model)[:2] == (1.0, "exact_match")
    assert ws.score_50_50_v2(None, "peppermint", comb, model)[:2] == (0.0, "no_response")
    s, m, _ = ws.score_50_50_v2("mint", "peppermint", comb, model)
    assert m == "combined_50_50" and 0.3 < s < 1.0
    # a response unrelated to the odor falls through to one of the fallback branches
    assert ws.score_50_50_v2("zzzqqq", "oregano", comb, model)[1] in ("semantic_only", "wrong_odor", "no_match")


def test_all_subjects_scored_and_counts():
    _, _, expd = profiles()
    model = StubModel()
    comb = ws.build_combined_profiles_50_50_v2(expd, model)
    sd = yaml.safe_load(open(os.path.join(ROOT, "config", "free_responses.yaml")))
    df = ws.score_all_subjects(sd, comb, model)
    assert len(df) == 51 and df.groupby("odor").size().to_dict() == {"grapefruit": 17, "oregano": 17, "peppermint": 17}
    assert (df.loc[df.method == "no_response", "similarity"] == 0).all() and df["similarity"].between(0, 1).all()
    dc = ws.descriptor_counts(sd)
    assert dc["n_responses"] == 51 and dc["n_with_descriptor"] == 44
    assert dc["mean_descriptors"] == pytest.approx(52 / 44) and dc["pct_exactly_one_descriptor"] == pytest.approx(100 * 36 / 44)


def test_permutation_signflip_and_ttest_agree_on_a_clear_effect():
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.1, 17)
    b = a - 0.3 + rng.normal(0, 0.05, 17)
    df = pd.DataFrame({"odor": ["peppermint"] * 17 + ["oregano"] * 17 + ["grapefruit"] * 17,
                       "similarity": np.r_[a, b, b - 0.05]})
    t = ws.behavior_tests(df, n_permutations=2000)
    row = t[(t.odor_1 == "peppermint") & (t.odor_2 == "oregano")].iloc[0]
    assert row["p_permutation_unpaired"] < 0.01 and row["p_ttest_paired"] < 0.001 and row["p_signflip_paired_exact"] < 0.001
    assert ws.signflip_paired_p(np.zeros(5), np.zeros(5)) == 1.0
    # exact sign-flip test on 3 pairs: 8 patterns, the observed one and its mirror -> p = 2/8
    assert ws.signflip_paired_p([3, 3, 3], [0, 0, 0]) == pytest.approx(0.25)


def test_neural_behavior_correlation_shapes():
    rows = []
    rng = np.random.default_rng(1)
    for s in ["22", "23", "24", "26"]:
        for o in ["peppermint", "oregano", "citrus"]:
            rows += [{"subject": s, "odor": o, "label": 0, "y_proba": rng.random()}, {"subject": s, "odor": o, "label": 1, "y_proba": rng.random()}]
    neural = ws.neural_gap_table(pd.DataFrame(rows))
    df = pd.DataFrame([{"subject": f"subject_{s}", "odor": o, "similarity": rng.random()} for s in ["22", "23", "24", "26"]
                       for o in ["peppermint", "grapefruit", "oregano"]])
    merged, out = ws.neural_behavior_correlation(df, neural)
    assert len(merged) == 12 and [r["odor"] for r in out] == ["peppermint", "oregano", "citrus", "pooled"] and out[-1]["n"] == 12


def test_script_end_to_end_with_a_fake_sentence_transformers(tmp_path):
    fake = tmp_path / "fake"
    (fake / "sentence_transformers").mkdir(parents=True)
    (fake / "sentence_transformers" / "__init__.py").write_text(
        "import sys\nsys.path.insert(0, %r)\nfrom test_word_scoring import StubModel\n__version__ = 'fake'\n"
        "def SentenceTransformer(name, revision=None):\n    return StubModel()\n" % os.path.dirname(__file__))
    rows = []
    rng = np.random.default_rng(2)
    for s in [22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
        for o in ["peppermint", "oregano", "citrus"]:
            rows += [{"subject": s, "odor": o, "label": 0, "y_proba": rng.random()}, {"subject": s, "odor": o, "label": 1, "y_proba": rng.random()}]
    pd.DataFrame(rows).to_csv(tmp_path / "neural.csv", index=False)
    params = {"word_scoring": {"responses": os.path.join(ROOT, "config", "free_responses.yaml"),
                               "goodscents_selected": os.path.join(ROOT, "config", "goodscents_selected.csv"), "goodscents_ref": "main",
                               "model": "stub", "model_revision": None, "n_perm": 200, "seed": 0}, "paper": {}, "checks_enabled": False}
    yaml.safe_dump(params, open(tmp_path / "params.yaml", "w"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(fake), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_word_scoring.py"), "--params", str(tmp_path / "params.yaml"),
                        "--out-dir", str(tmp_path / "out"), "--fig-dir", str(tmp_path / "fig"), "--neural-predictions", str(tmp_path / "neural.csv")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    out = tmp_path / "out"
    for f in ["scores_by_response.csv", "behavior_summary.csv", "behavior_tests.csv", "si31_neural_behavior_correlation.csv", "combined_profiles.json"]:
        assert (out / f).exists(), f
    assert (tmp_path / "fig" / "fig4CD_recognition.pdf").exists()
    assert len(pd.read_csv(out / "scores_by_response.csv")) == 51
