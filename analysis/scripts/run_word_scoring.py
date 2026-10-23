#!/usr/bin/env python
"""Free-response odor identification (Section 1.6, Fig. 4C/D, SI 3.1): GoodScents descriptor frequency + word-embedding similarity.

Replaces the preprocessing notebook's cells 21-27 and 29-30.  Needs `sentence-transformers` (see envs/embeddings.yml) and, the first time, internet
access to download the model; GoodScents is read from the deposited subset file (no internet) unless --goodscents-dir / --download is given.

    python scripts/run_word_scoring.py --params results_paper/word_scoring/params.yaml --out-dir results_paper/word_scoring \
        --fig-dir figures_paper --neural-predictions results_paper/classification/neural_predictions_early.csv
"""
import argparse
import contextlib
import io
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

from olfaction_eeg import word_scoring as ws
from olfaction_eeg.checks import Checks
from olfaction_eeg.fig4cd import make_fig4cd

warnings.filterwarnings("ignore")


def load_model(name, revision=None):
    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers is not installed in this environment (pip install sentence-transformers); see envs/embeddings.yml")
    model = SentenceTransformer(name, revision=revision) if revision else SentenceTransformer(name)
    info = {"model": name, "requested_revision": revision, "sentence_transformers": sentence_transformers.__version__}
    try:
        import torch, transformers
        info.update({"torch": torch.__version__, "transformers": transformers.__version__})
    except Exception:
        pass
    commit = None
    try:                                               # older transformers keep the hub commit id on the model config
        commit = getattr(model[0].auto_model.config, "_commit_hash", None)
    except Exception:
        pass
    if not commit:
        try:                                           # otherwise the name of the cached snapshot folder is the commit id
            from huggingface_hub import snapshot_download
            path = snapshot_download(name if "/" in name else f"sentence-transformers/{name}", revision=revision, local_files_only=True)
            commit = os.path.basename(os.path.normpath(path))
        except Exception:
            pass
    info["model_commit_hash"] = commit
    return model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--neural-predictions", default=None, help="neural_predictions_early.csv of the classification stage (for SI 3.1)")
    ap.add_argument("--goodscents-dir", default=None, help="folder with pyrfume-data goodscents molecules/behavior/stimuli csv (instead of the subset file)")
    ap.add_argument("--download", action="store_true", help="read GoodScents from pyrfume-data on GitHub (see word_scoring.goodscents_ref)")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    cfg, paper = p["word_scoring"], (p.get("paper") or {}).get("word_scoring", {})
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)
    log = io.StringIO()

    subjects_data = yaml.safe_load(open(cfg["responses"]))
    # ---- GoodScents: deposited subset (default), a local copy, or a download
    with contextlib.redirect_stdout(log):
        if a.goodscents_dir or a.download:
            gs = ws.load_goodscents(a.goodscents_dir, ref=cfg["goodscents_ref"])
            found = ws.search_all_datasets_fixed({"goodscents": gs})
            ws.selected_table(found).to_csv(os.path.join(a.out_dir, "goodscents_selected_rebuilt.csv"), index=False)
        else:
            found = ws.found_from_selected(pd.read_csv(cfg["goodscents_selected"]))
        empirical_profiles = ws.build_empirical_scoring_from_data({"goodscents": True}, found)
        empirical_profiles_expanded = ws.expand_profiles_with_synonyms(empirical_profiles)
    model, env_info = load_model(cfg["model"], cfg.get("model_revision"))
    with contextlib.redirect_stdout(log):
        combined_profiles_v2 = ws.build_combined_profiles_50_50_v2(empirical_profiles_expanded, model)
    df_v2 = ws.score_all_subjects(subjects_data, combined_profiles_v2, model)
    open(os.path.join(a.out_dir, "scoring_log.txt"), "w").write(log.getvalue())
    json.dump(env_info, open(os.path.join(a.out_dir, "embedding_versions.json"), "w"), indent=2)
    df_v2.to_csv(os.path.join(a.out_dir, "scores_by_response.csv"), index=False)
    json.dump({o: {k: float(v) for k, v in prof.items()} for o, prof in combined_profiles_v2.items()},
              open(os.path.join(a.out_dir, "combined_profiles.json"), "w"), indent=1)

    summ = ws.odor_summary(df_v2)
    summ.to_csv(os.path.join(a.out_dir, "behavior_summary.csv"))
    tests = ws.behavior_tests(df_v2, n_permutations=cfg["n_perm"], seed=cfg["seed"])
    tests.to_csv(os.path.join(a.out_dir, "behavior_tests.csv"), index=False)
    dc = ws.descriptor_counts(subjects_data)
    print("\nmodel / library versions:", env_info)
    print("\n=== recognition score by odor ===\n" + summ.round(3).to_string())
    print(f"overall: {df_v2['similarity'].mean():.3f} +- {df_v2['similarity'].std():.3f}; >=0.70: {(df_v2['similarity'] >= 0.70).sum()}/{len(df_v2)}; "
          f">=0.50: {(df_v2['similarity'] >= 0.50).sum()}/{len(df_v2)}")
    print("\n=== pairwise tests ===\n" + tests.round(4).to_string(index=False))
    print("\ndescriptor counts (SI 3.1):", {k: round(v, 3) if isinstance(v, float) else v for k, v in dc.items()})

    # ---- Fig. 4C / D (the example of panel C: "mint" for peppermint)
    emp_mint = empirical_profiles_expanded.get("peppermint", {}).get("mint", 0.0)
    v1, v2 = (model.encode(t, convert_to_numpy=True).reshape(1, -1) for t in ("mint", "peppermint"))
    from sklearn.metrics.pairwise import cosine_similarity
    cos_mint = float(cosine_similarity(v1, v2)[0, 0])
    make_fig4cd(df_v2, combined_profiles_v2, empirical_profiles_expanded, model, out_prefix=os.path.join(a.fig_dir, "fig4CD_recognition"))
    print(f"\nFig. 4C example 'mint' -> peppermint: empirical {emp_mint:.3f}, cosine {cos_mint:.3f}, score {0.5 * emp_mint + 0.5 * cos_mint:.3f}")

    # ---- SI 3.1
    corr_rows = []
    if a.neural_predictions and os.path.exists(a.neural_predictions):
        neural_df = ws.neural_gap_table(pd.read_csv(a.neural_predictions))
        merged, corr_rows = ws.neural_behavior_correlation(df_v2, neural_df)
        pd.DataFrame(corr_rows).to_csv(os.path.join(a.out_dir, "si31_neural_behavior_correlation.csv"), index=False)
        print("\n=== SI 3.1: Spearman, neural decision gap vs behavioral score ===\n" + pd.DataFrame(corr_rows).round(3).to_string(index=False))
    else:
        print("\n(SI 3.1 skipped: pass --neural-predictions)")

    # ---- checks
    checks = Checks(enabled=bool(p.get("checks_enabled", True) and paper))
    if checks.enabled:
        tol = paper.get("tol", 0.0006)
        for o, want in (paper.get("odor_mean_sd") or {}).items():
            checks.number(f"{o}: mean recognition score", summ.loc[o, "mean"], want[0], tol)
            checks.number(f"{o}: SD", summ.loc[o, "std"], want[1], tol)
            checks.number(f"{o}: n subjects", summ.loc[o, "count"], 17, 0)
        ov = paper.get("overall") or {}
        checks.number("overall mean", df_v2["similarity"].mean(), ov.get("mean"), tol)
        checks.number("overall SD", df_v2["similarity"].std(), ov.get("sd"), tol)
        checks.number("responses with score >= 0.70", (df_v2["similarity"] >= 0.70).sum(), ov.get("n_ge_070"), 0)
        checks.number("responses with score >= 0.50", (df_v2["similarity"] >= 0.50).sum(), ov.get("n_ge_050"), 0)
        for key, want in (paper.get("permutation_p") or {}).items():
            o1, o2 = key.split("_vs_")
            r = tests[(tests.odor_1 == o1) & (tests.odor_2 == o2)].iloc[0]
            checks.number(f"{o1} vs {o2}: mean difference", r["mean_diff"], want["diff"], tol)
            checks.number(f"{o1} vs {o2}: permutation p", r["p_permutation_unpaired"], want["p"], tol)
        for key, want in (paper.get("paired_t") or {}).items():
            o1, o2 = key.split("_vs_")
            r = tests[(tests.odor_1 == o1) & (tests.odor_2 == o2)].iloc[0]
            checks.number(f"{o1} vs {o2}: paired t", r["t_paired"], want["t"], 0.0006)
            checks.number(f"{o1} vs {o2}: paired p", r["p_ttest_paired"], want["p"], 0.0006)
            checks.number(f"{o1} vs {o2}: paired Cohen's d", r["cohens_d_paired"], want["d"], 0.0006)
        c4 = paper.get("fig4c") or {}
        checks.number("Fig 4C: 'mint' empirical frequency", emp_mint, c4.get("empirical"), tol)
        checks.number("Fig 4C: 'mint' cosine similarity", cos_mint, c4.get("cosine"), tol)
        checks.number("Fig 4C: 'mint' final score", 0.5 * emp_mint + 0.5 * cos_mint, c4.get("final"), tol)
        for k, want in (paper.get("descriptor_counts") or {}).items():
            checks.number(f"SI 3.1 {k}", dc[k], want, 0.05 if "pct" in k else 0.005)
        for r in corr_rows:
            want = (paper.get("si31") or {}).get(r["odor"])
            if want:
                checks.number(f"SI 3.1 {r['odor']}: n", r["n"], want["n"], 0)
                checks.number(f"SI 3.1 {r['odor']}: Spearman r", r["spearman_r"], want["r"], 0.0006)
                checks.number(f"SI 3.1 {r['odor']}: p", r["p"], want["p"], 0.0006)
        for odor, want in (paper.get("empirical_profile") or {}).items():
            prof = empirical_profiles[odor]
            checks.number(f"GoodScents profile {odor}: unique descriptors", len(prof), want["n_unique"], 0)
            for d, v in (want.get("top") or {}).items():
                checks.number(f"GoodScents profile {odor}: '{d}'", prof.get(d, np.nan), v, 0.0006)
        for desc, want in (paper.get("combined_profile_peppermint") or {}).items():
            checks.number(f"combined profile peppermint: '{desc}'", combined_profiles_v2["peppermint"].get(desc, np.nan), want, tol)
    checks.report(os.path.join(a.out_dir, "word_scoring_checks.csv"), strict=a.strict)


if __name__ == "__main__":
    main()
