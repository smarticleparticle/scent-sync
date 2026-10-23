"""Free-response odor identification score (paper Section 1.6, Fig. 4C/D, SI 3.1) -- hybrid of GoodScents descriptor frequency and
word-embedding similarity.

Ported from the preprocessing notebook (epochs_eeg_respiration_withtimestamps.ipynb) cells 21 (responses), 22 (permutation test), 23-24
(GoodScents search, empirical profiles, synonym expansion) and 25 (all-mpnet-base-v2 profiles and the scoring rule).  Functions below are the
notebook's, extracted from its source and unchanged; the glue is new: ``load_goodscents`` (cell 23, GoodScents part only -- Leffingwell and Keller were
loaded but never used), ``found_from_selected`` / ``selected_table`` (offline subset of GoodScents), ``score_all_subjects`` (the loop of cell 25),
the statistics (cells 26, 29-30) and the descriptor counts of SI 3.1.

NOTE on the GoodScents search (cell 24): the notebook appends only the first three matches of every search term to the molecule list
(``matches.head(3)``), in the order of the file.  That is what the published profiles are built from (peppermint: 8 molecules, grapefruit: 3,
oregano: 4); it is reproduced here and recorded in ``goodscents_selected.csv``.
"""
import numpy as np
import pandas as pd
from collections import Counter  # noqa: F401  (used by the notebook's functions)
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401  (used by the notebook's functions)

ODORS = ["peppermint", "grapefruit", "oregano"]
PYRFUME_RAW = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/{ref}/goodscents/{name}.csv"


def load_goodscents(source=None, ref="main"):
    """Merged GoodScents table (notebook cell 23): behavior (descriptors) + stimuli + molecules, columns include ``name`` and ``Descriptors``.

    ``source``: a folder containing molecules.csv / behavior.csv / stimuli.csv, or None to download them from pyrfume-data at ``ref``.
    """
    def read(name):
        if source:
            return pd.read_csv(f"{source}/{name}.csv")
        return pd.read_csv(PYRFUME_RAW.format(ref=ref, name=name))
    gs_molecules, gs_behavior, gs_stimuli = read("molecules"), read("behavior"), read("stimuli")
    gs_full = gs_behavior.merge(gs_stimuli, on='Stimulus')
    return gs_full.merge(gs_molecules, on='CID')


# ---------------------------------------------------------------------------------------------------------
# notebook cell 24, verbatim (search_all_datasets_fixed prints what it finds)
# ---------------------------------------------------------------------------------------------------------
def search_all_datasets_fixed(pyrfume_data):
    """
    Search all loaded datasets for menthol, limonene, carvacrol/thymol
    """
    
    search_terms = {
        'peppermint': ['menthol', 'mint', 'peppermint', 'menthone'],
        'grapefruit': ['limonene', 'citrus', 'grapefruit'],
        'oregano': ['carvacrol', 'thymol', 'oregano']
    }
    
    found_molecules = {odor: [] for odor in search_terms}
    
    if 'goodscents' in pyrfume_data:
        print("\n" + "="*70)
        print("SEARCHING GOODSCENTS")
        print("="*70)
        
        gs = pyrfume_data['goodscents']
        
        for odor, terms in search_terms.items():
            for term in terms:
                matches = gs[gs['name'].str.contains(term, case=False, na=False)]
                
                if len(matches) > 0:
                    print(f"\n{odor.upper()} - '{term}': {len(matches)} matches")
                    for _, row in matches.head(3).iterrows():
                        print(f"  • {row['name']}")
                        if 'Descriptors' in row:
                            print(f"    Descriptors: {row['Descriptors']}")
                        
                        found_molecules[odor].append({
                            'source': 'goodscents',
                            'name': row['name'],
                            'cid': row.get('CID', None),
                            'descriptors': row.get('Descriptors', '')
                        })
    
    return found_molecules


def build_empirical_scoring_from_data(pyrfume_data, found_molecules):
    """
    Calculate actual similarity scores from GoodScents descriptor co-occurrence
    """
    
    if 'goodscents' not in pyrfume_data:
        print("ERROR: GoodScents data not available")
        return None
    
    empirical_profiles = {}
    
    for odor, molecule_list in found_molecules.items():
        print(f"\n{'='*70}")
        print(f"BUILDING PROFILE FOR {odor.upper()}")
        print(f"{'='*70}")
        
        all_descriptors = []
        molecule_count = 0
        
        for mol in molecule_list:
            if mol['source'] == 'goodscents' and 'descriptors' in mol:
                desc_string = mol['descriptors']
                
                if pd.notna(desc_string):
                    descriptors = [d.strip().lower() for d in str(desc_string).replace(';', ',').split(',')]
                    descriptors = [d for d in descriptors if len(d) > 0]
                    
                    all_descriptors.extend(descriptors)
                    molecule_count += 1
        
        from collections import Counter
        descriptor_counts = Counter(all_descriptors)
        
        print(f"  Total descriptors: {len(all_descriptors)}")
        print(f"  Unique descriptors: {len(descriptor_counts)}")
        print(f"  From {molecule_count} molecules")
        
        profile = {}
        for desc, count in descriptor_counts.items():
            normalized_score = count / molecule_count if molecule_count > 0 else 0
            normalized_score = min(normalized_score, 1.0)
            profile[desc] = normalized_score
        
        sorted_profile = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n  TOP 15 DESCRIPTORS:")
        for desc, score in sorted_profile[:15]:
            print(f"    {desc:30s}: {score:.3f}")
        
        empirical_profiles[odor] = profile
    
    return empirical_profiles


# ---------------------------------------------------------------------------------------------------------
# notebook cell 24, verbatim
# ---------------------------------------------------------------------------------------------------------
def expand_profiles_with_synonyms(empirical_profiles):
    """
    Add common synonyms to profiles
    """
    
    synonym_rules = {
        'mint': ['minty', 'peppermint'],
        'minty': ['mint'],
        'menthol': ['mentholic'],
        'cool': ['cooling'],
        'cooling': ['cool'],
        'citrus': ['citrusy'],
        'fruity': ['fruit'],
        'herb': ['herbal'],
        'herbal': ['herb'],
        'spice': ['spicy'],
        'spicy': ['spice'],
        'wood': ['woody'],
        'woody': ['wood'],
    }
    
    expanded_profiles = {}
    
    for odor, profile in empirical_profiles.items():
        expanded = profile.copy()
        
        for desc, score in profile.items():
            if desc in synonym_rules:
                for syn in synonym_rules[desc]:
                    if syn not in expanded:
                        expanded[syn] = score
        
        expanded_profiles[odor] = expanded
    
    return expanded_profiles


# ---------------------------------------------------------------------------------------------------------
# notebook cell 25, verbatim
# ---------------------------------------------------------------------------------------------------------
def build_combined_profiles_50_50_v2(empirical_profiles, semantic_model):
    """
    50/50 combination with BETTER semantic model
    """
    
    combined_profiles = {}
    
    for odor in ['peppermint', 'grapefruit', 'oregano']:
        print(f"\n{'='*70}")
        print(f"BUILDING PROFILE FOR {odor.upper()}")
        print(f"{'='*70}")
        
        profile = {}
        empirical = empirical_profiles.get(odor, {})
        
        # Process empirical descriptors
        sorted_empirical = sorted(empirical.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop descriptors from GoodScents:")
        for desc, freq in sorted_empirical[:10]:
            print(f"  {desc:25s}: {freq:.3f}")
        
        print(f"\nCombining GoodScents (50%) + Semantic Similarity (50%):")
        
        for desc, emp_freq in sorted_empirical[:30]:
            # Compute cosine similarity
            try:
                desc_vec = semantic_model.encode(desc, convert_to_numpy=True).reshape(1, -1)
                odor_vec = semantic_model.encode(odor, convert_to_numpy=True).reshape(1, -1)
                cos_sim = cosine_similarity(desc_vec, odor_vec)[0, 0]
                cos_sim = max(0.0, float(cos_sim))
            except:
                cos_sim = 0.0
            
            # 50/50 combination
            final_score = 0.5 * emp_freq + 0.5 * cos_sim
            
            profile[desc] = final_score
            
            if final_score > 0.3:
                print(f"  {desc:25s}: {final_score:.3f} = 0.5×{emp_freq:.3f} + 0.5×{cos_sim:.3f}")
        
        # Add additional semantic terms
        additional_terms = {
            'peppermint': ['mint', 'minty', 'peppermint', 'cool', 'cooling', 'toothpaste', 'gum'],
            'grapefruit': ['grapefruit', 'citrus', 'fruity', 'sour', 'pink', 'tangy'],
            'oregano': ['oregano', 'herb', 'herbal', 'spicy', 'pizza', 'italian', 'savory']
        }
        
        print(f"\nAdding semantic-only terms:")
        for term in additional_terms.get(odor, []):
            if term not in profile:
                try:
                    term_vec = semantic_model.encode(term, convert_to_numpy=True).reshape(1, -1)
                    odor_vec = semantic_model.encode(odor, convert_to_numpy=True).reshape(1, -1)
                    cos_sim = cosine_similarity(term_vec, odor_vec)[0, 0]
                    cos_sim = max(0.0, float(cos_sim))
                except:
                    cos_sim = 0.0
                
                # No empirical data, so emp_freq = 0
                final_score = 0.5 * 0.0 + 0.5 * cos_sim
                
                if final_score > 0.25:
                    profile[term] = final_score
                    print(f"  {term:25s}: {final_score:.3f} = 0.5×0.000 + 0.5×{cos_sim:.3f}")
        
        combined_profiles[odor] = profile
    
    return combined_profiles


def score_50_50_v2(descriptor, actual_odor, combined_profiles, semantic_model):
    """
    Score using 50/50 combination with better model
    """
    
    if descriptor is None or pd.isna(descriptor):
        return 0.0, 'no_response', ''
    
    descriptor = str(descriptor).lower().strip()
    
    # Exact match
    if descriptor == actual_odor:
        return 1.0, 'exact_match', 'Perfect identification'
    
    # Check profile
    profile = combined_profiles.get(actual_odor, {})
    
    if descriptor in profile:
        score = profile[descriptor]
        
        if score >= 0.70:
            interpretation = 'Excellent'
        elif score >= 0.50:
            interpretation = 'Good'
        elif score >= 0.30:
            interpretation = 'Moderate'
        else:
            interpretation = 'Weak'
        
        return (score, 'combined_50_50', interpretation)
    
    # Not in profile - compute on the fly
    try:
        desc_vec = semantic_model.encode(descriptor, convert_to_numpy=True).reshape(1, -1)
        odor_vec = semantic_model.encode(actual_odor, convert_to_numpy=True).reshape(1, -1)
        cos_sim = cosine_similarity(desc_vec, odor_vec)[0, 0]
        cos_sim = max(0.0, float(cos_sim))
        
        # No empirical data, so score = 0.5 * cos_sim
        score = 0.5 * cos_sim
        
        if score > 0.20:
            return (score, 'semantic_only', f"Semantic: {cos_sim:.3f}")
    except:
        pass
    
    # Check cross-category
    for other_odor, other_profile in combined_profiles.items():
        if other_odor != actual_odor and descriptor in other_profile:
            return (0.15, 'wrong_odor', f"Valid for {other_odor}")
    
    return (0.05, 'no_match', 'No match')


# ---------------------------------------------------------------------------------------------------------
# notebook cell 22, verbatim
# ---------------------------------------------------------------------------------------------------------
def permutation_test_meandiff(scores1, scores2, n_permutations=10000, seed=0):
    rng = np.random.default_rng(seed)
    observed_diff = scores1.mean() - scores2.mean()

    pooled = np.concatenate([scores1, scores2])
    n1 = len(scores1)

    null_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        null_diffs[i] = shuffled[:n1].mean() - shuffled[n1:].mean()

    p_value = (np.abs(null_diffs) >= np.abs(observed_diff)).mean()
    return observed_diff, p_value


# ---------------------------------------------------------------------------------------------------------
# offline subset of GoodScents: exactly the molecules the search above puts into the profiles
# ---------------------------------------------------------------------------------------------------------
def selected_table(found_molecules):
    """DataFrame (odor, term order, name, CID, Descriptors) of the molecules in ``found_molecules`` -- deposited as goodscents_selected.csv."""
    rows = []
    for odor, mols in found_molecules.items():
        for k, m in enumerate(mols):
            rows.append({"odor": odor, "rank": k, "name": m["name"], "cid": m["cid"], "descriptors": m["descriptors"]})
    return pd.DataFrame(rows)


def found_from_selected(df):
    """Inverse of ``selected_table``: the ``found_molecules`` structure used by ``build_empirical_scoring_from_data``."""
    found = {odor: [] for odor in ["peppermint", "grapefruit", "oregano"]}
    for _, r in df.sort_values(["odor", "rank"]).iterrows():
        found[r["odor"]].append({"source": "goodscents", "name": r["name"], "cid": r["cid"],
                                 "descriptors": r["descriptors"] if pd.notna(r["descriptors"]) else ""})
    return found


# ---------------------------------------------------------------------------------------------------------
# scoring loop of notebook cell 25 (same statements, wrapped in a function)
# ---------------------------------------------------------------------------------------------------------
def score_all_subjects(subjects_data, combined_profiles_v2, semantic_model_v2):
    results_v2 = []

    for subject_id, responses in subjects_data.items():
        for odor in ['peppermint', 'grapefruit', 'oregano']:
            descriptors = responses.get(odor, [])
        
            if descriptors and descriptors[0] is not None:
                similarities = []
                methods = []
                explanations = []
            
                for desc in descriptors:
                    sim, method, explanation = score_50_50_v2(
                        desc, odor, combined_profiles_v2, semantic_model_v2
                    )
                    similarities.append(sim)
                    methods.append(method)
                    explanations.append(explanation)
            
                best_idx = similarities.index(max(similarities))
            
                results_v2.append({
                    'subject': subject_id,
                    'odor': odor,
                    'descriptors': ', '.join([str(d) for d in descriptors if d]),
                    'best_descriptor': descriptors[best_idx],
                    'similarity': similarities[best_idx],
                    'method': methods[best_idx],
                    'explanation': explanations[best_idx]
                })
            else:
                results_v2.append({
                    'subject': subject_id,
                    'odor': odor,
                    'descriptors': 'none',
                    'best_descriptor': None,
                    'similarity': 0.0,
                    'method': 'no_response',
                    'explanation': ''
                })

    return pd.DataFrame(results_v2)


# ---------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------
PAIRS = [('peppermint', 'oregano'), ('peppermint', 'grapefruit'), ('oregano', 'grapefruit')]


def odor_score_arrays(df_v2):
    """Per-odor arrays of the 17 subject scores, in subject order (the ``odor_scores`` of notebook cell 22, which was not defined in the saved notebook)."""
    return {o: df_v2[df_v2['odor'] == o]['similarity'].values for o in ODORS}


def signflip_paired_p(a, b):
    """Exact two-sided sign-flip permutation p of the paired mean difference (all 2^n sign patterns; n <= 20)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    signs = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1)
    null = (signs * d).mean(axis=1)
    return float((np.abs(null) >= abs(d.mean()) - 1e-15).mean())


def behavior_tests(df_v2, n_permutations=10000, seed=0):
    """Pairwise comparisons: the unpaired permutation test of notebook cell 22 (quoted in the text), the paired t-tests of cells 26/27 (the stars of
    Fig. 4D), and an exact paired sign-flip test (added: the same subjects contribute a score for every odor)."""
    scores = odor_score_arrays(df_v2)
    rows = []
    for o1, o2 in PAIRS:
        s1, s2 = scores[o1], scores[o2]
        diff, p_perm = permutation_test_meandiff(s1, s2, n_permutations=n_permutations, seed=seed)
        t, p_t = stats.ttest_rel(s1, s2)
        d = s1 - s2
        rows.append({"odor_1": o1, "odor_2": o2, "n": len(s1), "mean_diff": diff, "p_permutation_unpaired": p_perm,
                     "t_paired": t, "df": len(s1) - 1, "p_ttest_paired": p_t, "cohens_d_paired": np.mean(d) / np.std(d, ddof=1),
                     "p_signflip_paired_exact": signflip_paired_p(s1, s2)})
    return pd.DataFrame(rows)


def odor_summary(df_v2):
    g = df_v2.groupby('odor')['similarity'].agg(['count', 'mean', 'std', 'min', 'max'])
    return g.reindex(ODORS)


def descriptor_counts(subjects_data):
    """SI 3.1: descriptors per response (responses with at least one descriptor) and the share with exactly one descriptor."""
    n = [len([x for x in dd if x is not None]) for r in subjects_data.values() for dd in r.values()]
    with_any = [x for x in n if x > 0]
    return {"n_responses": len(n), "n_with_descriptor": len(with_any), "mean_descriptors": float(np.mean(with_any)),
            "pct_exactly_one_descriptor": float(100 * np.mean([x == 1 for x in with_any]))}


# ---------------------------------------------------------------------------------------------------------
# SI 3.1: neural decision-value gap vs behavioral score (notebook cells 29-30)
# ---------------------------------------------------------------------------------------------------------
def neural_gap_table(preds):
    """cell 29: per subject and odor, predicted odor probability minus predicted water probability (``neural_predictions_early.csv``)."""
    preds = preds.copy()
    preds['subject'] = preds['subject'].astype(str)
    rows = []
    for (subj, odor), group in preds.groupby(['subject', 'odor']):
        water_prob = group.loc[group['label'] == 0, 'y_proba']
        odor_prob = group.loc[group['label'] == 1, 'y_proba']
        if len(water_prob) == 0 or len(odor_prob) == 0:
            continue
        rows.append({'subject': subj, 'odor': odor, 'decision_gap': odor_prob.values[0] - water_prob.values[0]})
    return pd.DataFrame(rows)


def neural_behavior_correlation(df_v2, neural_df):
    """cell 30: Spearman correlations, per odor and pooled over subject-odor pairs.  Returns (merged table, rows)."""
    from scipy.stats import spearmanr
    behavioral_df = df_v2[['subject', 'odor', 'similarity']].rename(columns={'similarity': 'behavioral_score'})
    behavioral_df['odor'] = behavioral_df['odor'].replace({'grapefruit': 'citrus'})
    behavioral_df['subject'] = behavioral_df['subject'].astype(str).str.replace('subject_', '', regex=False)
    merged = behavioral_df.merge(neural_df, on=['subject', 'odor'], how='inner')
    rows = []
    for odor in ['peppermint', 'oregano', 'citrus']:
        sub = merged[merged['odor'] == odor]
        if len(sub) < 3:
            continue
        r, p = spearmanr(sub['behavioral_score'], sub['decision_gap'])
        rows.append({"odor": odor, "n": len(sub), "spearman_r": float(r), "p": float(p)})
    r, p = spearmanr(merged['behavioral_score'], merged['decision_gap'])
    rows.append({"odor": "pooled", "n": len(merged), "spearman_r": float(r), "p": float(p)})
    return merged, rows
