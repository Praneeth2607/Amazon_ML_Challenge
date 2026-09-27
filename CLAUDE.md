## AMAZON ML CHALLENGE ##

Static project facts. Anything that changes (runs, scores, next steps) lives in **STATUS.md**.

## 1. The task
- 3 sources of business records: `entity_id`, `business_name`, `business_address`, `country`.
- **Source 1** = clean master list. For each S1 record, list every **S2/S3** record that is the same business (0 to many).
- Train countries: US and India. **Test adds France** (unseen), so never hard-code countries.
- Output `matching_results.tsv`: `source1_entity_id<TAB>matched_entity_ids` (comma list, empty if none), one row per test S1.
- Also `candidate_pairs.tsv` (the shortlist the model scored). It's not scored but goes in the final zip.
- **Metric:** F0.5 per S1 record, then averaged. Precision counts 2× recall. Correct empty = 1.0, any wrong guess on a no-match = 0.

## 2. Rules (from ps.pdf + guidelines PDF)
- **No external data, APIs, geocoding or registries** (disqualification). Hand-written cleaning rules are fine.
- Model must be MIT/Apache-2.0 and ≤ 8B parameters (LightGBM = MIT ✓).
- Window: 25 Sep 2026 00:00 to 27 Sep 2026 23:59 IST. **Max 5 submissions per day.**
- Public leaderboard = part of test rows, visible now. Private = the rest, revealed at the end. Shortlisting uses **both**.
- Top 100 teams submit: methodology doc (fill `Documentation_template.md`), commented code, both TSVs.
- Validate before every upload: `utils/validate_submission.py`.

## 3. Dataset facts (checked on the real files)
| File | Rows | Notes |
|---|---|---|
| train_source1 | 2,206,821 | US 1.32M, India 0.88M |
| train_source2 | 5,034,616 | 15% non-Latin names (Hindi, Telugu, Gujarati…), website-style names |
| train_source3 | 5,285,603 | full state names ("Texas"), Indian state codes ("KA") |
| train_ground_truth | 2,206,821 | 7.64M links; avg 3.5 matches per S1 |
| test_source1 | 1,732,544 | India 810k, US 663k, France 259k |
| test_source2 | 4,887,273 | incl. France 703k |
| test_source3 | 5,082,316 | incl. France 732k |
- Only **5.6%** of S1 have no match. Each S2/S3 record belongs to **at most one** S1.
- Addresses carry most of the signal. They contain junk such as `null`, `<NULL>`, `N/A`.

## 4. Folders and files (`/mnt/project-files/`)
| Path | What it is |
|---|---|
| `ML/ps.pdf` | Official problem statement |
| `6ab10eb3b23ba_student_resource/student_resource/` | README (spec), `Documentation_template.md`, `utils/validate_submission.py` |
| `train_*.tsv`, `test_source1.tsv` | Uploaded data (root) |
| `*_part_N.tsv` | train_source3, test_source2, test_source3 split into parts (Windows split: BOM + CRLF; code handles it) |
| `code/erpipe.py` | **Core pipeline v1**: loading, cleaning, shortlist (`Index`), `features`, `macro_f05`, `decide` |
| `code/train_run.py` | v1 training (150k S1 sample vs all S2+S3, LightGBM, threshold search) |
| `code/predict_run.py` | v1 test prediction; saves each batch to `parts/` (resumable), writes both TSVs |
| `code/erpipe2.py`, `code/train_v2.py` | **v2 (not run on real data yet)**: stage-2 model using links between candidates |
| `code/merge_parts.py` | Joins `_part_N.tsv` files |
| `models/v1/` | `model.txt` (LightGBM), `model_meta.json` (threshold 0.65, feature list), `train_log.txt` |
| `guides/entity_resolution_kaggle.ipynb` | **Kaggle notebook v1.1** (= erpipe + train_run + predict_run + validator) |
| `guides/approach_explained_simply.md` | Plain-language explanation of approach, scoring, leaderboard |
| `guides/ml_challenge_explained.md` | First overview of task and files |
| `guides/todo.md` | **Outdated** (pre-data draft). Do not use its code. |

## 5. Kaggle setup
- Dataset `er-challenge-data` (private) at `/kaggle/input/datasets/jaatinnn/er-challenge-data/student_resource/dataset/{train,test}`.
- Notebook finds files automatically under `/kaggle/input`. Settings: CPU (no accelerator), Internet On.
- Run with **Save Version → Save & Run All (Commit)**. It runs on Kaggle's servers, so the laptop can be off.
- End of a successful run: `done: 1732544 rows …` then `PASS — no blocking issues found`.
- Download `output/matching_results.tsv` from the version's Output tab.

## 6. Current approach (v1)
1. **Clean:** transliterate every script to Latin letters (`unidecode`), lowercase, strip punctuation, `&`→and, drop website parts (.com/www),
   expand abbreviations (rd→road, st→street, r→rue…), drop `null`/`N/A`.
   Names also get a **skeleton** (no vowels, ph→f, double letters merged, 0→o, 1→l) so "limited" and "limittedd" both become `lmtd`.
   Legal words (Pvt, Ltd, LLC, Inc, SARL…) are removed from the core name.
2. **State detection:** US + Indian state names and codes, including transliterated ones; Telangana is treated as Andhra Pradesh.
3. **Shortlist (blocking):** per country and per state, TF-IDF top-15 by name (char 3-grams of skeleton)
   + top-15 by address (words and word pairs). Records with no state go in a country-wide pool (top-10).
   France has no states, so it uses its whole country.
4. **Features (~36):** rapidfuzz token-set/sort, ratio, partial, Jaro-Winkler on names, skeletons and addresses;
   house-number overlap; same state; lengths; non-Latin flag; rank and gap vs the S1's other candidates.
5. **Model:** LightGBM (330 trees, 255 leaves), trained on 150k random train S1 (seed 0) vs all S2+S3 records.
6. **Decide:** keep a pair if p ≥ 0.65, and give each S2/S3 record only to its best S1.

## 7. Working agreements with Jatin
- Jatin is new to ML. Explain in plain language and give exact click-by-click steps.
- Only upload a new version when **our validation F0.5** clearly beats the last one. Never tune on the public board.
- Code must be tested on real data before it is handed over. Always run the official validator.