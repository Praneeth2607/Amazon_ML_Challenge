# Business Entity Resolution -- Amazon ML Challenge 2026

Matches Source 2/3 business records to their Source 1 record across US, India and France,
using only the competition's own data (no external data, APIs, geocoding or registries).

## Final submission: v20

v20 is a weighted vote over six independently-produced `matching_results.tsv` files -- ours
(v9, v12) and teammates' (v5, v7, v8, v19, held and submitted separately by them) -- not a
single model. It:
1. Keeps every match proposed by **v19** (a teammate's independent model) unconditionally.
2. Adds any *extra* match that a weighted majority of the other five models agree on, with
   France (an unseen-in-training country) needing a stricter majority.

Public leaderboard score: **0.969** (confirmed). See the project's `STATUS.md` and
`Documentation_template.md` for the full history of how this was reached, including a failed
8-model attempt and the vote-mechanism bug this design fixes.

## Structure -- every version tried, in order

```
src/
  legacy/  er_solution.py          -- first from-scratch script; superseded, lower shortlist
                                       recall than v1, never scored
  v9/      er_v9_kaggle.py         -- core pipeline: keys shortlist + v8 features, LightGBM,
                                       per-country cut-offs, France check. Val F0.5 0.9753
  v10/     er_v10_kaggle.py        -- two-stage: v8 features + exact checks, stage 2 over all
                                       candidates of a business. Val F0.5 0.9730
  v11/     er_v11_kaggle.py        -- v9 + stage 2 without retraining. Never run
  v12/     er_v12_kaggle.py        -- v9's stage-1 + two-stage design (out-of-fold LightGBM,
                                       3-model stage 2). Val F0.5 0.9760 -- our other real model
                                       in v20
  v15/     er_v15_kaggle.py        -- v9 + ordinal/zero-pad cleaning fixes + a 3rd "name +
                                       house-number" shortlist search. Val F0.5 0.9763
  v16/     er_v16_kaggle.py        -- v15's stage-1 + v12's two-stage design. Val F0.5 0.9767
                                       -- our best single validated model
  v17/     er_v17_kaggle.py        -- v9 unchanged + word-by-word name-check features
                                       (one-word-changed / initials detection). Val F0.5 0.9762
  v18/     er_v18_kaggle.py        -- v17's features + v12's two-stage design. Val F0.5 0.9764
  merge/   er_v14_merge_kaggle.py  -- combines finished versions' answers into one (the v20 step)
```

Each `src/<version>/er_v*_kaggle.py` is a **standalone** script: paste the whole file into one
Kaggle notebook cell (CPU is enough for every version here). Every script installs its own
pinned dependencies at the top. `kaggle_notebooks_all_versions/` and `new_notebooks/` (project
root, one level up) hold teammates' own earlier/parallel model notebooks (v1-v8, an
independent v10, a 105-feature "v14-model", and "v2.4") for reference.

## How to reproduce v20 end-to-end

1. **Train + predict v9 and v12** on Kaggle: each script's top has a `RUN = "train"` /
   `"predict,combine,check"` pair of sessions (see the comment block at the top of each file
   for exact settings and timing -- training and prediction together are close to Kaggle's
   12-hour session limit, so they're deliberately split into two sessions, with the second
   attaching the first's Notebook Output).
2. **Get v5, v7, v8, v19's `matching_results.tsv`** from the teammates who built them.
3. **Rename** each finished file after its version (`v5.tsv`, `v9.tsv`, `v12.tsv`, ... -- see
   the docstring at the top of `src/merge/er_v14_merge_kaggle.py`) and upload them all to one
   private Kaggle dataset.
4. **Run the merge**: paste `src/merge/er_v14_merge_kaggle.py` into a new notebook cell with
   that dataset and the competition data attached, and run it (CPU, ~1 minute). Its defaults
   are already set for v20 (`ALWAYS_KEEP="v19"`, `SKIP="v16,v17,v18"` -- i.e. only v5/v7/v8/v9/v12
   vote, v19 is always kept; the SKIP list is a leftover from an earlier, abandoned 8-model
   experiment and has no effect here since those files aren't part of this dataset). It writes
   `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the official
   validator automatically.

## Requirements

See `requirements.txt` (pinned). Install with `pip install -r requirements.txt`, or let each
script install its own dependencies (they do this automatically via `pip install -q` at the top).
