Pristine copies of the files that `scripts/finalize_release.py` edits (README, CITATION.cff, DEPOSIT.md, environment.yml, the GoodScents source note), taken **before**
the release placeholders were filled in. `tests/test_finalize.py` runs the script on these copies, so the tests keep working after the real files have been finalized.
If you change the placeholder wording in the real files before releasing, change it here too (`test_placeholders_match_the_script_patterns` tells you).
