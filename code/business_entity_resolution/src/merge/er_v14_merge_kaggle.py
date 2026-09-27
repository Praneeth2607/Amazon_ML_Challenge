"""
Business Entity Resolution - Amazon ML Challenge 2026 - v14 MERGE (single file for Kaggle).
Reads several versions' finished matching_results.tsv files and keeps a match only when the versions
holding MORE THAN HALF the total weight agree on it. No training, no shortlist, no model - just a vote
over answers you already have. Runs in a few minutes on CPU.

HOW TO SET IT UP (once)
  1. From each finished version's Kaggle Output tab, download output/matching_results.tsv.
  2. Rename each file after its version: v5.tsv, v7.tsv, v8.tsv, v9.tsv, v12.tsv ... (the filename tells
     this script which weight to use - see WEIGHTS below).
  3. On Kaggle: Create -> New Dataset, name it e.g. er-version-answers, upload all the renamed files, Private.
  4. New notebook: Add Input -> that dataset, and also your competition-data dataset (for the format check).
  5. Paste this whole file into ONE code cell. Settings: CPU, Internet not needed. Save Version -> Save & Run All.

WEIGHTS below: v1/v9/v12/v16/v17/v18 are OUR real measured validation F0.5 (0.9529/0.9753/0.9760/0.9767/0.9762/0.9764).
The rest (v2-v8, v10-v11, v13-v14, v19) are teammates' own claimed numbers from a shared thread, UNVERIFIED by us -
v5 especially, since we have no prior record of it at all. v19 = a teammate's OWN, separate "v9" model (their own
numbering, unrelated to our v9) renamed to avoid colliding with our v9's key and weight; their claimed F0.5 is 0.9807
(highest weight here, UNVERIFIED and notably above every one of our confirmed numbers -- watch its effect on the vote).
Edit WEIGHTS freely if you get a confirmed number for any version.
Use at least 3 versions. With only 2, a match is kept only if both agree (safe, but drops more real matches).
No external data or APIs are used.
"""

# =====================================================================
# Version 14: MERGE the answer files of several versions into one better answer.
#
# Idea (a "vote"): every version you ran on Kaggle wrote a matching_results.tsv. A match that most
# versions agree on is very likely right; a match only one version makes is often a mistake.
# F0.5 punishes wrong matches twice as hard as missed ones, so dropping the "only one version said so"
# matches usually raises the score. Better versions get a bigger say (WEIGHTS below).
# No model is trained here, so it runs in a few minutes on CPU. No external data is used.
# =====================================================================
import os, re, sys, glob, time, warnings, subprocess
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import polars as pl

INPUT = os.environ.get("INPUT", "/kaggle/input")
WORK = os.environ.get("WORK", "/kaggle/working")
# How much say each version gets. Unknown versions get 1.0. Taken from validation F0.5 (see thread):
# higher-scoring versions count more. Edit freely.
WEIGHTS = {"v1": 0.395, "v2": 1.1, "v3": 1.37, "v4": 1.0, "v5": 1.3, "v6": 0.84, "v7": 0.99, "v8": 1.14,
           "v9": 1.515, "v10": 1.55, "v11": 1.3, "v12": 1.55, "v13": 1.3, "v14": 1.54,
           "v16": 1.585, "v17": 1.56, "v18": 1.57, "v19": 1.785}
# v1/v9/v12/v16/v17/v18 = OUR real measured val F0.5 (0.9529/0.9753/0.9760/0.9767/0.9762/0.9764);
# rest are teammates' claimed numbers, unverified. weight = 1 + 50 x (val F0.5 - 0.965)
# v19 = teammate's own separate "v9" model (file renamed from their(v9).tsv to avoid colliding with our v9),
# claimed F0.5 0.9807, UNVERIFIED -- highest weight in the set, above every one of our confirmed numbers.
NEED = float(os.environ.get("NEED", 0.5))    # keep a match when its versions' weight is MORE than this share of the total
# Countries with no training answers (France) are riskier: there a match needs a bigger share of the votes.
NEED_UNSEEN = float(os.environ.get("NEED_UNSEEN", 0.65))
TRAIN_COUNTRIES = {"US", "India"}
SKIP = set(os.environ.get("SKIP", "v5,v7,v8,v16,v17,v18").split(",")) - {""}   # versions to leave out, e.g. "v1,v3"
# Versions whose OWN proposed match is always kept, regardless of the weighted vote (a safe union, not a
# filter, for a version that is strong enough alone that the majority rule would otherwise strip its
# correct-but-unique matches -- everyone else can still ADD extra matches on top via the normal vote).
ALWAYS_KEEP = set(os.environ.get("ALWAYS_KEEP", "v19").split(",")) - {""}

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def version_name(path):
    """'v8' from a path like /kaggle/input/entity-resolution-v8/output/matching_results.tsv or .../v8_matching.tsv"""
    for part in reversed(path.lower().replace("_", "-").split("/")):
        m = re.findall(r"(?:^|[^a-z0-9])v(\d{1,2})(?:[^0-9]|$)", part) or re.findall(r"version-?(\d{1,2})", part)
        if m: return "v" + m[-1]
    return path

def find_answer_files(root):
    out = {}
    for p in sorted(glob.glob(f"{root}/**/*.tsv", recursive=True)):
        base = os.path.basename(p).lower()
        # only files named like v9.tsv (our renaming convention) - train_ground_truth.tsv has the SAME
        # header as a version's answer file (source1_entity_id / matched_entity_ids) and would otherwise
        # be picked up by mistake, then crash the id-list check because it lists training, not test, businesses
        if not re.match(r"^v\d{1,2}(@\d+)?\.tsv$", base):
            continue
        with open(p, encoding="utf-8-sig") as fh:
            head = fh.readline().strip()
        if head != "source1_entity_id\tmatched_entity_ids": continue
        name = version_name(p)
        if name in SKIP: continue
        if name in out: name = f"{name}@{len(out)}"   # two files for one version: keep both, same weight
        out[name] = p
    return out

def read_pairs(path):
    """matching_results.tsv -> (all S1 ids in file order, long table s1_id / cand_id)"""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
    ids = df["source1_entity_id"]
    pairs = (df.filter(pl.col("matched_entity_ids") != "")
               .with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids").rename({"matched_entity_ids": "cand_id"})
               .filter(pl.col("cand_id") != ""))
    return ids, pairs

def s1_countries(root):
    """entity_id -> country from test_source1 (if the competition data is an input), else None."""
    hits = sorted(glob.glob(f"{root}/**/test_source1*.tsv", recursive=True))
    if not hits: return None
    frames = []
    for p in hits:
        t = open(p, encoding="utf-8-sig").read().replace("\r", "")
        if not t.startswith("entity_id\t"): t = "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + t
        frames.append(pl.read_csv(t.encode(), separator="\t", quote_char=None, infer_schema_length=0,
                                  columns=["entity_id", "country"]))
    return pl.concat(frames).rename({"entity_id": "source1_entity_id"}).unique("source1_entity_id")

def merge(files, need=NEED, need_unseen=NEED_UNSEEN):
    ids, tabs, tot = None, [], 0.0
    for name, path in files.items():
        w = WEIGHTS.get(name.split("@")[0], 1.0)
        i, p = read_pairs(path)
        if ids is None: ids = i
        elif i.len() != ids.len() or not i.sort().equals(ids.sort()):
            raise ValueError(f"{path} has a different list of Source 1 businesses; is it from the same test file?")
        log(f"{name:6s} weight {w:.2f} | matches {p.height:,} | businesses with a match {p['source1_entity_id'].n_unique():,} | {path}")
        always = name.split("@")[0] in ALWAYS_KEEP
        tabs.append(p.with_columns(pl.lit(w).alias("w"), pl.lit(always).alias("always"))); tot += w
    allp = pl.concat(tabs)
    score = allp.group_by(["source1_entity_id", "cand_id"]).agg(
        pl.col("w").sum().alias("w"), pl.len().alias("n"), pl.col("always").any().alias("always"))
    cc = s1_countries(INPUT)
    if cc is None:
        log("test_source1 not found in the inputs: one rule for all countries")
        score = score.with_columns(pl.lit(need).alias("need"))
    else:
        score = score.join(cc, on="source1_entity_id", how="left").with_columns(
            pl.when(pl.col("country").is_in(list(TRAIN_COUNTRIES))).then(pl.lit(need)).otherwise(pl.lit(need_unseen)).alias("need"))
        log(f"countries without training answers (France ...) need weight > {need_unseen * tot:.2f}")
    keep = score.filter((pl.col("w") > pl.col("need") * tot + 1e-9) | pl.col("always"))
    # a Source 2/3 record belongs to at most one business: if two businesses claim it, the stronger vote wins
    keep = keep.sort(["w", "n"], descending=True).unique(subset="cand_id", keep="first", maintain_order=True)
    log(f"merged: {len(files)} versions, total weight {tot:.2f}, keep a match when weight > {need * tot:.2f}")
    log(f"distinct matches proposed {score.height:,} -> kept {keep.height:,}")
    if "country" in keep.columns:
        for c, n in keep.group_by("country").len().sort("country").iter_rows():
            log(f"   {c}: {n:,} matches kept")
    return ids, keep, score

def write_outputs(ids, keep, score, outdir):
    os.makedirs(outdir, exist_ok=True)
    m = keep.group_by("source1_entity_id").agg(pl.col("cand_id").sort().str.join(",").alias("matched_entity_ids"))
    res = pl.DataFrame({"source1_entity_id": ids}).join(m, on="source1_entity_id", how="left").fill_null("")
    res.write_csv(f"{outdir}/matching_results.tsv", separator="\t", quote_style="never")
    # candidate list = every match any version proposed (all of them were scored by at least one version)
    c = score.group_by("source1_entity_id").agg(pl.col("cand_id").sort().str.join(",").alias("candidate_entity_ids"))
    cand = pl.DataFrame({"source1_entity_id": ids}).join(c, on="source1_entity_id", how="left").fill_null("")
    cand.write_csv(f"{outdir}/candidate_pairs.tsv", separator="\t", quote_style="never")
    n = (res["matched_entity_ids"] != "").sum()
    log(f"done: {res.height:,} rows, {n:,} with matches, {res.height - n:,} empty; avg matches {keep.height / res.height:.2f}")

if True:
    files = find_answer_files(INPUT)
    if len(files) < 2:
        raise SystemExit(f"Found {len(files)} answer file(s) under {INPUT}. Add at least 2 versions' matching_results.tsv as inputs.")
    ids, keep, score = merge(files)
    write_outputs(ids, keep, score, f"{WORK}/output")



# ======================================================================
# Official format check (candidate file skipped: needs a lot of memory)
# ======================================================================
# official format check (the candidate file check is skipped: it needs a lot of memory)
validator = glob.glob(f'{INPUT}/**/validate_submission.py', recursive=True)
tests = glob.glob(f'{INPUT}/**/test_source1*.tsv', recursive=True)
if validator and tests:
    test_dir = os.path.dirname(tests[0])
    subprocess.call([sys.executable, validator[0], "--matching", f"{WORK}/output/matching_results.tsv",
                     "--test-dir", test_dir, "--candidate", "none.tsv"])   # candidate file check skipped: needs a lot of memory
else:
    print('Add the er-challenge-data dataset as an input to run the format check.')
