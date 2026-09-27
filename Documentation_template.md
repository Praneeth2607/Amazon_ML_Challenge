# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [fill in]
**Team Members:** Jatin D, Nityasri A, Praneeth VP (SRM University)
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary

Our final submission (v20) is a weighted vote over six independently-built models rather than
a single classifier: it keeps every match from our strongest model unconditionally, and lets
five other models only add matches they independently corroborate. This design grew out of
discovering that a plain majority vote silently discards a strong model's unique correct
matches, and it improved our public F0.5 from 0.966 to 0.969.

---

## 2. Methodology

### 2.1 Problem Analysis

Addresses carry most of the matching signal but contain junk (`null`, `<NULL>`, `N/A`).
15% of Source 2 names are non-Latin script (Hindi, Telugu, Gujarati, ...) and some are
website-style names. Source 3 uses full state names ("Texas") and Indian state codes ("KA");
Telangana is treated as Andhra Pradesh. Only 5.6% of Source 1 businesses have no true match, so
the class balance strongly favours finding matches, and each Source 2/3 record belongs to at
most one Source 1 business. France, present only in the test set, cannot be validated directly.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier, ensembled by a weighted vote across model variants.
**Core Innovation:** Instead of picking one trained model, we combine several finished answer
sets with a vote that is asymmetric on purpose: our best-performing model's matches are always
kept, and other models can only add extra, independently-corroborated matches on top -- never
remove from the strongest model's output. A plain symmetric weighted vote cannot let a single
strong model's *unique* matches through no matter its weight (its weight would need to exceed
the combined weight of every other model), so it silently strips a strong model's correct,
non-duplicated matches. Fixing this (`ALWAYS_KEEP` in `er_v14_merge_kaggle.py`) was the single
change that moved us from 0.966 to 0.969 on the public leaderboard.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** transliteration to Latin script + lowercasing + punctuation
  normalisation; a name "skeleton" (vowels dropped, ph->f, doubled letters merged, 0->o, 1->l)
  so spelling variants collapse together; legal-form words (Pvt, Ltd, LLC, Inc, ...) stripped
  from the core name; US/Indian state detection (including transliterated forms); per
  country-and-state TF-IDF shortlist, top-15 by name (character 3-grams of the skeleton) and
  top-15 by address (words and word pairs); records with no detected state fall into a
  country-wide pool. France has no states, so it blocks on the whole country.
- **Candidate pairs generated:** roughly 92.6 per Source 1 business under our base pipeline
  (v9/v12); the final v20 candidate file is the union of every matched-or-considered pair
  across all six ensembled models' own candidate sets.
- **How true matches were not lost:** shortlist recall was measured directly against the
  30k-business held-out validation set (97.97% overall for v9; 96.7% India, 98.9% US), and a
  later shortlist search that adds name-piece + house/plot-number TF-IDF raised this to 98.46%.

---

## 4. Matching Model

**Features used (~36 in the base pipeline, more in later variants):**
- Name features: rapidfuzz token-set/sort ratio, partial ratio, Jaro-Winkler on names,
  name skeletons, and metaphone/rarity-weighted words; word-by-word checks that catch a
  single changed word (e.g. "Chandrraj Traders" vs "Chandrreon Traders") and
  initials/acronym names (e.g. "mc.com" for "Marquez, Conway & Associates").
- Address features: rapidfuzz ratios on cleaned addresses, house-number overlap, same-state flag.
- Other: record lengths, non-Latin-script flag, candidate rank and score gap versus a
  business's other candidates, name-frequency (commonness) features.

**Model type:** LightGBM (single-model variants) and a two-stage design (variants v10/v12/v16/
v18) where a first-stage LightGBM's out-of-fold probabilities feed a second stage that looks at
all of a business's candidates together and asks whether each one looks like the business's
other strong candidates, decided by a 3-model team (LightGBM + XGBoost + CatBoost).
**Threshold selection method:** per-country probability cut-offs tuned by F0.5 on the 30k-
business validation set (e.g. v9: India 0.675, US 0.700, France 0.688 -- the average of the
other two, since France cannot be validated directly).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation (30k held-out US+India businesses):**
  v9 0.9753, v12 0.9760, v17 0.9762, v18 0.9764, v16 0.9767 (our best single validated model).
- **F_0.5 Score, public leaderboard (final submission, v20):** **0.969**.
- **How v20 was built:** v20 = a weighted vote of v5, v7, v8, v9, v12 (models with real or
  claimed validation scores), with v19 (a teammate's independently-built model, claimed
  validation 0.9807) kept unconditionally rather than voted on. An earlier attempt that simply
  added three more of our own model family (v16/v17/v18) into the vote *lowered* the score
  (0.964 vs. the prior 0.966): those three share the same core pipeline and the same
  validation set, so they added correlated agreement rather than independent evidence, diluting
  the more genuinely different teammate models already in the vote. v19 alone, submitted
  without any ensembling, scored 0.968; the ALWAYS_KEEP fix then added a modest ~0.83% of
  further, independently-corroborated matches on top of v19's own set to reach 0.969.
- **Common false positives (wrong merges):** look-alike businesses distinguished only by a
  single changed word, or acronym/initials names -- specifically targeted by the word-by-word
  feature block in v17/v18.
- **Common false negatives (missed matches):** weaker in India than the US throughout (recall
  96.7% vs 98.9% on the base shortlist); this is the largest remaining, un-exploited lever.

---

## 6. Conclusion

Our strongest lever was not a single better model but a principled way to combine several: once
we found that plain vote-based merging silently discards a strong model's correct-but-unique
matches, protecting one trusted model's output and only ever *adding* corroborated matches on
top turned out to be the single change that beat every individual model and every naive
ensemble we tried. The main lesson for anyone extending this: agreement between models trained
on the same pipeline and the same validation set is not independent evidence, and a merge
mechanism has to be checked for what it silently throws away, not just what its output stats
look like.

---

## Appendix

### A. Code Artefacts

Runnable code ships under `code/business_entity_resolution/` (`src/` holds our own two model
versions -- v9 and v12 -- plus the final `merge/` step, with a top-level `README.md` and pinned
`requirements.txt`). Entry points: run `src/v9/er_v9_kaggle.py` and `src/v12/er_v12_kaggle.py`
on Kaggle (train, then predict) to produce each version's own `matching_results.tsv` /
`candidate_pairs.tsv`; then run `src/merge/er_v14_merge_kaggle.py` over all six versions'
finished answer files (ours plus teammates' v5, v7, v8, v19, submitted separately by them) to
produce the final `output/matching_results.tsv` and `output/candidate_pairs.tsv` (v20). See
that README for exact settings.

### B. Additional Results

v20 merge statistics: 5,832,718 matches kept across 1,635,995 of 1,732,544 businesses
(avg. 3.37 matches/business), versus v19 alone's 5,784,535 matches / 1,634,168 businesses --
confirming the merge added matches on top of v19 rather than replacing or diluting them.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and
technical depth.
