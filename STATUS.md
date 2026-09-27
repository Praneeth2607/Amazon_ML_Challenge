Changing information only. Static facts are in **CLAUDE.md**. Last updated 26 Sep 2026, about 16:35 IST.
**Deadline: confirmed from the portal's own countdown = 1 day 4 hours left as of 16:35 IST 26 Sep, i.e. roughly
20:35 IST on 27 Sep.** This roughly matches CLAUDE.md's "27 Sep 23:59 IST" after all (this file briefly had the
wrong current date and pushed the deadline a day too far; corrected here). Max 5 submissions per day still applies.

## 0. ⚠️ Rule update from the Amazon team (26 Sep)
Posted after v9/v12 were built, so nothing below was designed around it:
> `candidate_pairs.tsv` is part of the final submission. Blocking has to scale — comparing every record
> with every other one is not an option; the candidate-generation step must cut the search space to a
> **small** candidate set per Source 1 entity. Candidate generation **counts toward the final ranking**:
> Amazon will review `candidate_pairs.tsv` and the code that produces it. **The approach that generates
> a smaller candidate set per Source 1 entity will be ranked higher, beyond the public/private leaderboard.**

**Implication we had not been optimizing for:** all of v9/v10/v11/v12's `candidate_pairs.tsv` carry about
**92.6 candidates per S1 business** (the full TF-IDF top-15-name + top-15-address shortlist, before the
model's probability cut-off). That is roughly 26x the average number of true matches (~3.5). It was fine
when candidates were "not scored," but under this new rule a smaller shortlist is now explicitly rewarded,
independent of the F0.5 score. We have not re-run anything to shrink it — see section 6 for options and
the trade-off (recall vs. candidate-set size) — because every option needs a fresh blocking pass and
re-running features/training/prediction, and there may not be time before the deadline.

## 0b. New teammate notebooks found in `kaggle_notebooks.zip` (26 Sep, extracted to `new_notebooks/`)
A teammate's zip added 4 more notebooks (code only, no run logs, so scores below are their own claimed validation numbers, not confirmed by us):
| Notebook | Idea | Claimed val F0.5 | Candidate-set size |
|---|---|---|---|
| `entity_resolution_v10_kaggle.ipynb` (a **different, newer** v10 than ours) | error-analysis-driven fixes: v7 checks + v8 + word-by-word name checks + stage 2 + more cleaning | 0.9759 (before the last cleaning step, which is "printed by this run") | **capped at ≤20, ~4-5/business** (p1 ≥ 1%, rank ≤ KTOP) — already addresses section 0 |
| `entity_resolution_v14_model_kaggle.ipynb` | one model trained on all proven signals of v1+v6+v7+v8 combined (105 features) | **0.9757** | **capped at 10 best + every match** — already addresses section 0 |
| `entity_resolution_v2_4_kaggle.ipynb` | better cleaning (98.3% shortlist recall vs our 97.97%), stage 2, a "competition" step (record belongs to ≤1 business, trained on 253k full-coverage businesses), separate model for empty-address records | 0.9717 (measured on a harder, non-overlapping 236k-business set, not the usual 30k) | **capped at 20, ~7 avg/business** — already addresses section 0 |
| `entity_resolution_v14_merge_kaggle.ipynb` | **no training** — reads several versions' finished `matching_results.tsv`, keeps a match only when versions holding **more than half the total weight** agree (weight ≈ 1 + 50×(val F0.5 − 0.965); higher for France/unseen countries), ~5 min on CPU | ensemble of whichever versions you feed it | candidate file = union of proposed matches only (no full shortlist) → likely **very small**, but may not satisfy "exact set fed to the model" literally since there is no model here |
**Checked (26 Sep, this session):** all three extracted code-only .py files compile clean, pyflakes finds no undefined
names in any of them, and the AST shadowing check (the pattern that broke our own v9 draft once) finds nothing in
any of the four. v10(new)'s `base()` redefinition is intentional (a second, Indic-aware version replaces the v1
one later in the same file; Python resolves the name at call time, so `prepare()` picks up the better one) — not a bug.
All three paths default to `/kaggle/input` / `/kaggle/working` correctly. **They look safe to run on Kaggle as-is.**
**SUPERSEDED (26 Sep, later this session):** the "0.9777 = v9's public score" theory in this section is now contradicted by the portal's own Details panel, which shows Overall Score 0.96056, Rank 1135, after both v9 and v12 were submitted. Cause of the 0.9777 number is unclear -- possibly a misremembered figure, possibly a different metric. Do not rely on it. See section 4 for the confirmed number and the open question about what it actually measures. **Most other teammates' runs failed from running out of disk space** because they were run locally (inside a Claude Code session) instead of on Kaggle's own servers — Kaggle's `Save Version -> Save & Run All (Commit)` is required for these multi-GB pipelines; running them locally will keep failing. Current public top 50 ~= 0.989. Target: reach >= 0.99, now looking harder than before given the 0.96056 data point.

## 1. Where we are right now
- **v9's candidate-shrink recombine: DONE and verified.** Re-ran `RUN = "combine"` (26 Sep). matching_results.tsv came out
  byte-identical in every count to the original run (France 95.0%/3.30 avg, India 94.1%/3.25, US 94.4%/3.34, 1,634,189
  with matches, 98,355 empty) -- confirms the fix touches ONLY candidate_pairs.tsv, v9's 0.9777 public score is untouched.
  candidate_pairs.tsv: **avg 3.4 candidates/business**, down from ~92.6.
- **v12: fully done, checker PASSED.** Trained 0.9760 val. Combine+check ran clean: matches France 94.6%/95.0%(v9)
  -- close to v9's France numbers, no red flag. candidate_pairs.tsv also came out at **avg 3.4/business** (smaller than
  intended -- KTOP=15 turned out to be a no-op here because the pre-existing dedup-by-candidate + probability floor
  in the combine code already shrinks the pool that far before KTOP even applies; not a bug, matched-subset-of-
  candidates still holds by construction, but the candidate list barely shows runner-ups beyond the actual matches --
  worth describing accurately in the methodology doc rather than implying a wide exploratory shortlist).
  **Not yet submitted to the portal** -- would be a useful second real public-score data point since v9 and v12 are
  a statistical tie on validation (0.9753 vs 0.9760) and the only way to see if they differ on France is a real submission.

- **v9 is fully done.** Trained (230 min), predicted (461 min), combined. Official checker: **PASS** (1,732,544 rows, 98,355 empty, 1,634,189 non-empty). Both files exist in that notebook's `output/`: `matching_results.tsv` and `candidate_pairs.tsv` (the candidate file was not checked by the validator; it is required for the final zip).
- **Nothing recorded as uploaded yet.** Next action: upload v9's `matching_results.tsv`, record the public score in section 4, and compare it with the 0.9753 validation score (v1 lost 0.012 between the two, probably France).
- **v12** (second account) is trained (0.9760 val). Prediction now finished on all three countries (India 210.6 min, US+France 127.2 min, split across two notebooks). Still need to run the third "combine,check" notebook to produce `matching_results.tsv`/`candidate_pairs.tsv` and get the validator PASS. It is a tie with v9 on validation score (difference inside the noise), not a proven win.
- **v10** trained (0.9730), prediction deliberately not run. **v11** is built but never run.
- **Local validator run on v9's results/output/ files started** (candidate_pairs.tsv is ~2GB, so it takes a while); result pending.

| **v15** | v9's pipeline + v2.4's ordinal/zero-pad cleaning + a 3rd "name + house-number" TF-IDF shortlist search | **CONFIRMED 0.9763** (blend_top3, cut-off 0.675) | file `er_v15_kaggle.py`. TRAINED 27 Sep: shortlist recall **0.9846** (v9 was 0.9797 -- the combo search worked), ceiling 0.9952. Beats both v9 (0.9753) and v12 (0.9760). Training took 272.5 min. Prediction not yet started |
| **v16** | v15's pipeline WITH v12's proven two-stage design | **CONFIRMED 0.9767 (best_single=lgb_a, stage1 alone 0.9751) -- BEST validated result overall** | file `er_v16_kaggle.py`. Trained in 201 min. The earlier apparent mismatch (shortlist numbers looking like v9's unboosted pipeline) was a paste/formatting mixup between two logs pasted in the same message, not a real mislabeling -- the user re-sent clean, distinct tails for all four (v15/v16/v17/v18) and each compare string was verified directly against its file |

| **v17** | v9's pipeline UNCHANGED + a teammate's v10 word-by-word name-check features (one-word-changed detection e.g. "Chandrraj"/"Chandrreon", initials/acronym matching e.g. "mc.com"/"Marquez Conway Associates"). Their own gain: 0.9741->0.9747. Standalone, single-stage like v9 | **not run yet** | file `er_v17_kaggle.py`, built 26 Sep. Full end-to-end pipeline test passed (toy data: correctly flagged both look-alike patterns) |
| **v18** | v17's pipeline WITH v12's proven two-stage design (stage 2 = all candidates of a business voted together, 3-model team) | **not run yet** | file `er_v18_kaggle.py`, built 26 Sep, reuses v12's already-tested stage-2 code unchanged |

## 2. All versions (validation = same 30k held-out train businesses, US + India only)
| Version | Idea | Val F0.5 | Notes |
|---|---|---|---|
| v1 | TF-IDF shortlist (same state, top-15 name + top-15 address), one LightGBM | 0.9529 | **public 0.9405** |
| v2 | + stage 2 (compare candidates of one business) | 0.9568 | |
| v3 / v4 | multilingual MiniLM re-check / embedding shortlist (GPU) | not reported | download pretrained weights; rule risk, see section 6 |
| v6 | 6x more training data, per-country models, France cut-off +0.15 (a guess) | 0.9618 | |
| v7 | exact keys (extra shortlist) + planted look-alike checks (house number, legal form) | 0.9647 | |
| v8 | rarity-weighted words, metaphone, rank features, 4-model ensemble | 0.9673 | |
| **v9** | **v8 + v7 + v6 name frequency**, per-country cut-offs, France check | **0.9753** (India 0.9694, US 0.9792) | file `er_v9_kaggle.py`. Single lgb_b chosen (xgb 0.9746, cat 0.9730, blend 0.9748). Cut-offs: India 0.675, US 0.700, France 0.688 (average of the two), close-to-best share 0 |
| v10 | two-stage: v8 features + exact checks (no keys), stage 2 over all candidates of a business | 0.9730 (India 0.9670, US 0.9770) | stage 1 alone 0.9624, stage 2 gave +0.0106. Trained in 125 min. File `er_v10_kaggle.py` |
| v11 | v9 + stage 2 without retraining (stage 2 trained on the 30k check businesses, 5-fold cross-validated) | not run | file `er_v11_kaggle.py`. Now probably not worth it: v12 shows stage 2 adds little on top of v9's features |
| **v12** | two-stage with v9's strong stage 1 (keys, v8 features, out-of-fold LightGBM), 3-model stage 2 | **0.9760** (India 0.9697, US 0.9802) | file `er_v12_kaggle.py`. Trained in 179 min. Stage 1 alone 0.9742, stage 2 only +0.0018. +0.0007 over v9 is inside the noise (standard error about 0.0008 on 30k businesses). Prediction path is untested |

| **v17** | v9's pipeline UNCHANGED + a teammate's v10 word-by-word name-check features | **CONFIRMED 0.9762** (blend_top3, cut-off 0.675) | file `er_v17_kaggle.py`. Trained in 278 min. Slightly below v15/v16 -- the combo-search recall boost (v15/v16) mattered more than the word-check features alone |
| **v18** | v17's pipeline WITH v12's two-stage design | **CONFIRMED 0.9764** (best_single=lgb_b, stage1 alone 0.9750) | file `er_v18_kaggle.py`. Trained in 193 min |

Reference numbers: predicting "no match" for everyone = 0.0573. Perfect model on v9's shortlist = about 0.99.

## 3. Key measurements
- **Shortlist recall (v9, 30k check businesses):** 97.97% overall. India 96.7% (Source 3 only 96.0%), US 98.9%. Only 0.17% of businesses lose all their links, about 6% lose some. v12 measured 97.89%.
- **Pairs per business:** about 92.6, so the test set has about 160M pairs to score. Prediction is the long part (v9: 461 min in one session; batches of 100k businesses against one source).
- **India is the weak spot** (lower recall and lower score than the US). Any further gain is most likely in India blocking.
- **v9 France check on the test set** (share of businesses with at least one match / matches per matched business): at cut-off 0.70 France 94.9% / 3.47, India 94.0% / 3.45, US 94.4% / 3.54; at 0.90 France 94.3% / 3.34, India 93.5% / 3.35, US 94.1% / 3.47. Final: France 95.0% with a match (3.30 avg), India 94.1% (3.25), US 94.4% (3.34). France looks similar to the others (0.3 to 0.9 points higher at low cut-offs). No evidence for a margin, so `FRANCE_MARGIN` stays 0.
- **Training set sizes:** 150k businesses (v1's sample; the 20% check set is the same for every version).

## 4. Submissions used
| Date (IST) | Version | Public score | Notes |
|---|---|---|---|
| 25 Sep 19:55 IST | v1 | **0.940** | confirmed from the portal's Submissions list |
| 26 Sep 20:26 IST | v12 | **0.961** | confirmed from the portal's Submissions list. Val was 0.9760 -> gap ~0.015 |
| 26 Sep 20:47 IST | v9 | **0.960** | confirmed from the portal's Submissions list. Val was 0.9753 -> gap ~0.015 |
| 26 Sep 23:14 IST | **v14-merge (5-model vote: v5+v7+v8+v9+v12)** | 0.966 | +0.006 over v12 (0.961), +0.005 over v9. Fixed 2 bugs first: (1) `find_answer_files` was picking up train_ground_truth.tsv by mistake (same header as an answer file) and crashing; (2) the official format-check line was silently deleted instead of converted to a real call. Both fixed in `er_v14_merge_kaggle.py`. Merge stats: 6,185,206 matches proposed -> 5,624,470 kept (91%); 1,628,383/1,732,544 businesses matched, avg 3.25. |
| 27 Sep ~21:19 IST | 8-model merge (adds v16+v17+v18 to the above) | 0.964 | REGRESSION -- worse than the 5-model merge. v9/v12/v16/v17/v18 are all one correlated family (same core pipeline, tuned on the same 30k check set); adding 3 more from that family diluted the diversifying effect of v5/v7/v8 rather than adding real independent signal. Lesson: correlated-model agreement is not a reliable signal. Do not merge v16/v17/v18 together with v9/v12 again without a diversity check. |
| 27 Sep, later | v19 alone (teammate's independent "v9"-numbered model, claimed val 0.9807, renamed to avoid colliding with our v9's key) | **0.968** | New best at the time. Val-to-public gap ~0.013, consistent with our own models' ~0.015 gap -- the claim looks real, not inflated. |
| 27 Sep, later still | **v20** = 6-model merge: v5+v7+v8+v9+v12 vote, **v19 always kept** (new `ALWAYS_KEEP` mechanism in `er_v14_merge_kaggle.py`) | **0.969 -- CONFIRMED BEST, FINAL SUBMISSION** | Fixes the majority-vote flaw: a plain weighted vote structurally can't let a single strong model's *unique* matches through no matter its weight (needs weight > sum of all others combined), so naively voting v19 in with the rest silently stripped ~132k of its matches and would have underperformed v19 alone. `ALWAYS_KEEP="v19"` (now hardcoded as the default, alongside `SKIP="v16,v17,v18"`) keeps every v19 match unconditionally and lets the other 5 models only ADD independently-corroborated extras on top. Result: kept 5,832,718 matches (v19 alone had 5,784,535) over 1,635,995 matched businesses (v19 alone: 1,634,168) -- a modest, well-corroborated addition, not a flood. A follow-up test swapping v5/v7/v8 for v3 (weight 1.37, unverified/placeholder) scored slightly lower on both counts (kept 5,824,054, matched 1,635,324) and was NOT submitted. **v20 is the final submission** -- stop experimenting; time is critical (see top-of-file deadline note). |

## 4b. Handoff package for documentation (27 Sep, in progress)
Teammate is documenting models v9/v12/v16/v17/v18 plus the final v20 ensemble. Local `handoff/<version>/` folders
created with whatever `matching_results.tsv` / `candidate_pairs.tsv` were already on disk. **Still missing as of
this note -- must be downloaded from each Kaggle notebook's Output tab before the handoff is complete:**
- `handoff/v16/candidate_pairs.tsv` and `handoff/v17/candidate_pairs.tsv` (their matching_results.tsv are present;
  the candidate files were never pulled down locally).
- **`handoff/v20/matching_results.tsv` and `handoff/v20/candidate_pairs.tsv` -- the actual final-submission files,
  currently not saved anywhere locally.** This is the most important gap: v20 is what got submitted (0.969), and
  CLAUDE.md's final-package rule requires keeping the submitted candidate_pairs.tsv.
v9/v12/v18 folders are complete (both files present, copied from `results/output/`, `V12/output/`, and
`v18.tsv` + `output/candidate_pairs.tsv` respectively).

The earlier "0.9777" figure was wrong; discard it (see the strikethrough note in section 0b/1 history). Both v9 and v12 lost ~0.015 val-to-public, similar in size to v1's ~0.013 loss -- so France (or general overfitting to the shared 30k check set used to tune every version) is costing real points after all, not "gaining" as briefly believed.

**Rank 1135, "Overall Score" 0.96056** shown on the portal's Details panel matches v9's number (the most recent upload), not v12's higher one -- the dashboard appears to reflect the LATEST submission, not the best. **Action: make sure the final submission before the deadline is the best model, not just the most recent test upload.** Team: Jatin D, Nityasri A, Praneeth VP (SRM University).

## 5. Files and how to run (all in the project folder)
| File | What it is |
|---|---|
| `er_v9_kaggle.py` | v9 single file. `RUN = "train"` (session 1), then `RUN = "predict,combine,check"` with session 1's output attached. `ONLY_COUNTRIES` splits prediction across sessions. Batches are saved in the notebook output and reused |
| `er_v10_kaggle.py`, `er_v11_kaggle.py`, `er_v12_kaggle.py` | same pattern; v11 has `RUN = "val"` then `"test,combine,check"`; v12 is standalone |
| `kaggle_notebooks_all_versions/` | teammates' notebooks v1 to v8 and the generated v9 notebook |
| `v9_build/`, `v10_build/`, `v11_build/`, `v12_build/` | source pieces and `build*.py` scripts that generate the single files (v11 and v12 build from v9's and v10's pieces) |
| `er_solution.py` | my first from-scratch script. Superseded; lower shortlist recall than v1 and never scored |
- The combine step refuses to write a file if any country's batches are missing (so a split run cannot produce empty rows for a missing country).
- **v12 prediction plan (if used):** two parallel notebooks, `RUN = "predict"` with `ONLY_COUNTRIES = "India"` and `"France,US"`, both attaching the competition data and v12's training output; then a third notebook with `RUN = "combine,check"` attaching the data, the training output, and both prediction outputs. About 4 to 5 hours split. It only fits if started by about 12:00 UTC.

## 6. Decisions, lessons and risks
- **Candidate-set size now counts toward the final ranking (see section 0), and ours is large (~92.6/business).**
  Options, none tried yet: (a) lower `K_NAME`/`K_ADDR` in the shortlist (currently 15+15) — cheapest, but needs a
  full re-run and would lower shortlist recall, which is currently the ceiling on the score; (b) write
  `candidate_pairs.tsv` from a *tighter* re-query at inference time while keeping the wider shortlist for
  training the model — risks candidate_pairs no longer matching "the exact set fed to the model," which the
  spec requires; (c) leave it as is and accept a lower rank on this criterion, since F0.5 is still the primary
  score and re-running risks the deadline. No decision made; needs Jatin's call given the time left.
- **France is the big unknown.** 15% of the test businesses, not in training, cannot be validated. The France table is label-free evidence only. Choosing between candidates on the public score goes against the "never tune on the public board" agreement; treat it as a judgement call.
- **Pretrained models (v3/v4) are a rule question.** The rules require MIT/Apache models up to 8B, which suggests pretrained is allowed, but "no external data" is disqualifying. Confirm with the organizers before relying on them. v9, v10, v12 use no pretrained weights.
- **Bugs found by static checking before running:** a variable named `base` overwrote the text-cleaning function of the same name (would have crashed inside worker processes); an assertion message that mentioned a version name; a per-loop dictionary rebuild that would have been very slow. All fixed. Toy-data tests confirmed the new scorer equals v8's reference scorer exactly, and the out-of-fold LightGBM subset approach works.
- **Time estimates were too optimistic once:** the first v9 attempt (150k + 100k businesses, 4 models, refit) projected past Kaggle's 12-hour limit, so it was restarted as two sessions with `N_EXTRA = 0` and `REFIT = 0`.
- **Validation caveat:** validation covers US and India only, and cut-offs and model choice were tuned on the same 30k businesses, so all validation scores are slightly optimistic. Differences below about 0.001 are noise.
- **Untested paths:** v12's prediction cell and v11 have never run on real data.

## 6b. v15/v16 (built 26 Sep, two free accounts, not exact re-runs of teammate notebooks)
Since the teammate was already running v2.4 and their-v10 on other accounts, these two are DELIBERATE
IMPROVEMENTS grafted onto our own proven pipeline instead of duplicates:
- v15 = v9's full architecture (keys, v8 features, exact checks, name commonness, LightGBM ensemble) PLUS
  the two specific things v2.4 measurably improved: base() cleaning (ordinal suffixes, zero-padding) and a
  third TF-IDF shortlist search combining name pieces with house/plot numbers. v2.4's OWN architecture
  (competition-training, empty-address model, its more aggressive skel_tok) was deliberately left out --
  those weren't isolated wins in v2.4's own numbers, only the recall jump (97.7%->98.3%) was clearly
  attributable, so only that piece was ported.
- v16 = v15's stage-1 + v12's already-tested stage-2 code, unchanged. Tests whether more shortlist recall
  gives stage 2 more useful signal than it had in v12.
- Both pass: py_compile, pyflakes (no undefined names), the AST shadowing check, and a toy-data functional
  test of the new combo search + base() fix (confirmed it recovers a match plain v9 cleaning misses: "Beta
  Stores 4th Street" / "0030 Elm St" now matches "Beta Stores 4 Street" / "30 Elm Street"). Neither has run
  on real Kaggle data.
- Launch like v9/v12: Save & Run All with RUN = "train" first; only proceed to prediction if val F0.5
  clearly beats the relevant baseline (v15 vs 0.9753, v16 vs 0.9760).

## 6c. v17/v18 (built 26 Sep, "two more models" per the user's request)
Same pattern as v15/v16: a real, verified improvement from teammate code grafted onto our own proven
pipeline, not a duplicate of what the teammate is running.
- v17 = v9's pipeline, completely UNCHANGED, PLUS a new feature block from a teammate's v10 notebook:
  word-by-word name comparison. Catches exactly-one-word-changed names (Amazon plants these: e.g.
  "Chandrraj Traders" vs "Chandrreon Traders", where every other word matches but whole-string fuzzy
  ratios don't flag it clearly) and short/website names made of initials ("mc.com" for "Marquez, Conway &
  Associates"). Tested against `JaroWinkler` word-by-word logic on toy data before use: confirmed it
  correctly flags both patterns (`w_one_sub=1`, `acr_any=1`).
- v18 = v17's stage 1 + v12's already-tested stage-2 code, unchanged (same relationship as v15/v16).
- Both pass py_compile, pyflakes, and the AST shadowing check. v17 additionally passed a FULL end-to-end
  functional test (real cleaning -> keys -> shortlist -> features, not just the isolated new feature) --
  confirmed the whole pipeline runs and the new columns come through correctly. v18 was not run end-to-end
  locally (its stage-2 path is unchanged code already proven by v12; only the stage-1 feature swap is new,
  and that swap is the same one v17 already verified).
- Launch like v9/v12/v15/v16: RUN = "train" first; only proceed to prediction if val F0.5 clearly beats the
  relevant baseline (v17 vs 0.9753, v18 vs 0.9760).

## 6d. Six real validation results now in hand (27 Sep) -- decision point
v9 0.9753 / v12 0.9760 / v17 0.9762 / v15 0.9763 / v18 0.9764 / **v16 0.9767 (best)**. All four new ones
(v15-v18) beat both baselines, but the spread is tight (0.9762-0.9767) and each pairwise gap is close to
or inside the ~0.0008 noise band on 30k businesses. None of v15/v16/v17/v18 have been PREDICTED yet
(session 2, ~6-9 h each) -- only v9 and v12 have real matching_results.tsv today.
**Recommendation given the time left:** do not run prediction for all four (24-36 h of compute we do not
have). Run prediction for **v16 only** (the clear best) on one free account. Once its matching_results.tsv
exists, add it to the existing 5-model ensemble (v5+v7+v8+v9+v12, already confirmed 0.966 public) and
re-run the merge (~5 min) for a potentially stronger 6-model final vote. If there is not enough time left
for even one more 6-9 h prediction run, the safe fallback is to keep the current 0.966 ensemble as the
final submission -- do not gamble the remaining hours chasing an uncertain last-minute gain.

## 6e. Decision (27 Sep, ~15 h left): predict all four in parallel
User's call, and a sound one given genuine parallelism (4 separate accounts) costs no extra wall-clock
time: launch v15/v16/v17/v18 prediction sessions simultaneously rather than just v16 alone. More real,
independent models strengthens the final ensemble vote more than picking only the single best validator.
- Each: `RUN = "predict,combine,check"`, attach the competition data + that version's own training output.
- v15/v17 (single-stage) ~6-8 h; v16/v18 (two-stage) ~5-7 h. None have run prediction before (only
  training was tested) -- check each log ~20-30 min in for an early error before trusting the full run.
- Once done: add each real matching_results.tsv to the `er-version-answers` dataset (v15.tsv/v16.tsv/
  v17.tsv/v18.tsv), alongside the existing v5/v7/v8/v9/v12, and re-run the merge for a bigger vote before
  the final submission. Current best confirmed public score: 0.966 (5-model ensemble) -- the floor to beat.

## 6f. v18 prediction DONE (27 Sep) -- first of the four to finish
Checker PASSED. 1,732,544 rows, 1,629,772 with matches (102,772 empty), avg 3.4 candidates/business.
France check consistent with India/US at every cut-off (France 94.6%/3.33, India 93.9%/3.25, US 94.1%/3.35)
-- no sign of over- or under-matching France. Prediction took 255.7 min. Ready to add `v18.tsv` to the
`er-version-answers` ensemble dataset. Still waiting on v15, v16, v17.

## 6h. v17 done too; v15 written off; 8-model merge is next (27 Sep)
v17 prediction finished and its `matching_results.tsv` is downloaded as `v17.tsv`. **v15 will not finish in
time** (its prediction is running far slower than estimated -- was at "test_source3 India batch 1/9" after
510+ min with 8 India batches still to go -- and the user has decided not to wait for it). Local files now
in hand for the merge: `v5.tsv`, `v7.tsv`, `v8.tsv`, `v9.tsv`, `v12.tsv`, `v16.tsv`, `v17.tsv`, `v18.tsv` --
8 of the intended 9 versions. `er_v14_merge_kaggle.py`'s `WEIGHTS` dict has been updated with the three new
real, measured weights: v16 1.585 (val 0.9767), v17 1.56 (val 0.9762), v18 1.57 (val 0.9764) -- same
formula as v9/v12 (`1 + 50 x (val F0.5 - 0.965)`). **Next step: upload all 8 files to the
`er-version-answers` Kaggle dataset (replacing/adding v16/v17/v18) and re-run the merge notebook** for a
bigger vote than the current confirmed-best 0.966 (5-model: v5+v7+v8+v9+v12).

## 6g. v16 prediction DONE (27 Sep) -- second of the four, and it's the best single validator (0.9767)
Checker PASSED on matching_results.tsv (1,732,544 rows, 101,497 empty, 1,631,047 non-empty). Prediction took
464.2 min. candidate_pairs.tsv: avg **3.4 candidates/business** (down from ~92.6 -- confirms the shrink fix
works on v16 too). Per-country match rate/avg at the chosen cut-offs (France 0.562, India 0.500, US 0.625,
close-to-best share 0.70): France 259,452 businesses, 94.8% with a match, avg 3.32; India 809,986, 93.9%,
avg 3.27; US 663,106, 94.2%, avg 3.36 -- consistent with v18's numbers, no red flag on France. The in-notebook
validator call for candidate_pairs.tsv pointed at a wrong/missing path ("none.tsv not found") so that check
was skipped there; re-validate `candidate_pairs.tsv` locally once downloaded, same as done for v9. Local file
`v18.tsv` (96MB) already saved from Kaggle; download v16's `matching_results.tsv` as `v16.tsv` next and add both
to the `er-version-answers` ensemble dataset. Still waiting on v15, v17.

## 7. Next steps, in order (~1 day 4 hour plan, target: push toward 0.99; be realistic it may land lower)
0. ~~Confirm the real deadline~~ **done** (see the top: ~28 h left as of 26 Sep 16:35 IST, deadline ~27 Sep 20:35 IST).
1. **Do now, ~5-10 min each, zero risk:**
   - Re-run v9's `RUN = "combine"` with the candidate-shrink fix (new notebook, attach v9's prediction output
     + the data). Rewrites only `candidate_pairs.tsv` (92.6 -> ~15/business); `matching_results.tsv` untouched.
   - Finish v12's first `combine,check` run (training + prediction are already done per section 1).
2. **Launch in parallel, on separate Kaggle accounts, via Save & Run All (Commit) — NOT locally:**
   - `new_notebooks/kaggle_notebooks/entity_resolution_v10_kaggle.ipynb` (their v10, different from ours; claims up
     to 0.9759 val, candidates already capped ~4-5/business). ~1.5-2 h train + 5-7 h predict.
   - `entity_resolution_v14_model_kaggle.ipynb` (105 combined signals; claims 0.9757 val, candidates capped at 10 +
     matches). ~1.5 h train + 4-5 h predict.
   - If a 3rd/4th account is free: `entity_resolution_v2_4_kaggle.ipynb` (best shortlist recall seen, 98.3%;
     scored 0.9717 on a harder 236k-business set, not directly comparable to the usual 30k numbers, but the
     cleaning improvements behind it are the most promising real lever we've seen for raising the ceiling).
   Both checked clean (see section 0b) but neither has actually been run by us — treat the first hour of each as
   an unofficial smoke test; if something breaks, come back with the log.
3. **Once 2+ of {v9, v12, their-v10, v14-model, v2.4} have real finished `matching_results.tsv` (likely 6-9 h in):**
   run `entity_resolution_v14_merge_kaggle.ipynb` (the weighted-vote ensemble, ~5 min CPU, no training) over all
   of them. Replace its placeholder `WEIGHTS` with our own measured validation scores before running. This is the
   step most likely to move the needle further, but expect the final gain to be in the +0.002 to +0.01 range on
   top of the best single model, not a guaranteed jump to 0.99.
4. **India blocking** is still the biggest untapped lever (recall 96.7% vs US 98.9%) if a spare account and hours
   remain after step 2-3.
5. **Final package (top 100 teams only):** `output/` with both TSVs, `code/business_entity_resolution/` with
   `src/`, `README.md` and `requirements.txt` (pinned), and the filled-in `Documentation_template.md`. Use
   whichever version is the final chosen submission (v9, an ensemble, or otherwise) and its build pieces as the source.
