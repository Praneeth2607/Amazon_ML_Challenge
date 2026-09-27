#!/usr/bin/env python3
"""
Business Entity Resolution - Amazon ML Challenge 2026 (single-file pipeline, "v3").

WHAT IT DOES
  Stage "train"   : learn a LightGBM pair classifier on a sample of training Source-1 records
                    (against ALL training S2+S3 records), tune the decision rule on held-out S1.
  Stage "predict" : shortlist candidates for every test S1 record, score them, write
                    output/matching_results.tsv and output/candidate_pairs.tsv, run the validator.

HOW TO RUN ON KAGGLE (CPU, Internet On, dataset attached)
  1) Smoke test first (about 20-30 min, tiny sample, writes to output_quick/):
         !python er_solution.py --stage all --quick
  2) Full run (Save Version -> Save & Run All (Commit)):
         !python er_solution.py --stage all
     or run the two stages separately:  --stage train   then   --stage predict

PIPELINE
  1. Clean   : transliterate every script to Latin (unidecode), lowercase, strip punctuation,
               drop website parts / titles / legal words, build a vowel-less "skeleton" of the name,
               drop null-like junk from addresses, expand address abbreviations, detect the state
               (US + India only; other countries such as France simply have no state).
  2. Block   : per country, per state (records with no state are added to every state's pool):
               TF-IDF (hashed) top-K by name skeleton char-3-grams + top-K by address words/pairs.
  3. Features: rapidfuzz similarities on name / skeleton / address, house-number and PIN overlap,
               lengths, script flag, TF-IDF similarities and rank/gap vs the S1's other candidates.
               No country is hard-coded as a feature, so unseen countries (France) work.
  4. Model   : LightGBM (MIT licence).
  5. Decide  : keep p >= threshold, give every S2/S3 record only to its best S1, and optionally keep
               only candidates within a fraction of the S1's best score (both tuned on validation).

No external data, APIs, geocoding or registries are used. Only the provided files.
"""
import os, sys, re, csv, glob, json, time, argparse, subprocess, functools
import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _ensure(mod, pip_name=None):
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or mod])


_ensure("unidecode")
_ensure("rapidfuzz")
_ensure("lightgbm")

from unidecode import unidecode
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
import lightgbm as lgb

# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------
CFG = dict(
    N_TRAIN=150_000, N_VAL=30_000, SEED=0,
    K_NAME=15, K_ADDR=15,              # shortlist size per channel
    NAME_MAX_DF=0.003, ADDR_MAX_DF=0.003,   # drop features present in more than this share of docs
    HASH_FEATS=2 ** 22,
    NUM_ROUNDS=800,
)
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f} min]", *a, flush=True)


def get_ctx():
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


# ----------------------------------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------------------------------
LEGAL_MAP = {"pvt": "private", "ltd": "limited", "corp": "corporation", "inc": "incorporated", "co": "company"}
LEGAL = {"private", "pvt", "limited", "ltd", "llc", "llp", "inc", "incorporated", "corp", "corporation",
         "co", "company", "sarl", "sas", "sa", "eurl", "sci", "gmbh", "plc", "lp", "opc", "pty"}
TITLES = {"mr", "mrs", "ms", "dr", "shri", "sri", "smt", "the"}

NONALNUM = re.compile(r"[^a-z0-9]+")
DOT_ACR = re.compile(r"(?<=\b[a-z])\.(?=[a-z]\b)")          # l.l.c. -> llc.
WEB_RE = re.compile(r"\bwww\.|\.(?:co\.in|com|net|org|in|co|fr|us|io|biz|info|edu|gov)\b")
JUNK = re.compile(r"<\s*null\s*>|\bnull\b|\bn/a\b|\bnan\b|\bnone\b")
TRAIL_DIG = re.compile(r"\s*\d+$")
VOW = re.compile("[aeiou]")
DBL = re.compile(r"(.)\1+")

ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "dr": "drive", "blvd": "boulevard",
    "bd": "boulevard", "bld": "boulevard", "ln": "lane", "ct": "court", "hwy": "highway", "pl": "place",
    "cir": "circle", "trl": "trail", "pkwy": "parkway", "ter": "terrace", "sq": "square", "chem": "chemin",
    "imp": "impasse", "nr": "near", "opp": "opposite", "blk": "block", "bldg": "building", "flr": "floor",
    "apt": "apartment", "ste": "suite", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "mt": "mount",
    "ft": "fort", "twp": "township", "twsp": "township", "cross": "cross",
}
R_RULE_NEXT = {"de", "du", "des", "la", "le", "d"}      # "R. DE DIEPPE" -> "rue de dieppe"

US_STATES = ("AL alabama;AK alaska;AZ arizona;AR arkansas;CA california;CO colorado;CT connecticut;"
             "DE delaware;DC district of columbia;FL florida;GA georgia;HI hawaii;ID idaho;IL illinois;"
             "IN indiana;IA iowa;KS kansas;KY kentucky;LA louisiana;ME maine;MD maryland;"
             "MA massachusetts;MI michigan;MN minnesota;MS mississippi;MO missouri;MT montana;"
             "NE nebraska;NV nevada;NH new hampshire;NJ new jersey;NM new mexico;NY new york;"
             "NC north carolina;ND north dakota;OH ohio;OK oklahoma;OR oregon;PA pennsylvania;"
             "RI rhode island;SC south carolina;SD south dakota;TN tennessee;TX texas;UT utah;"
             "VT vermont;VA virginia;WA washington;WV west virginia;WI wisconsin;WY wyoming;PR puerto rico")
INDIA_STATES = [
    "ap|andhra pradesh|telangana|ts|tg", "ar|arunachal pradesh", "as|assam", "br|bihar",
    "cg|chhattisgarh|chattisgarh|ct", "ga|goa", "gj|gujarat", "hr|haryana", "hp|himachal pradesh",
    "jh|jharkhand", "ka|karnataka", "kl|kerala", "mp|madhya pradesh", "mh|maharashtra", "mn|manipur",
    "ml|meghalaya", "mz|mizoram", "nl|nagaland", "od|odisha|orissa|or", "pb|punjab", "rj|rajasthan",
    "sk|sikkim", "tn|tamil nadu|tamilnadu", "tr|tripura", "up|uttar pradesh", "uk|uttarakhand|uttaranchal|ut",
    "wb|west bengal", "an|andaman and nicobar islands|andaman nicobar", "ch|chandigarh",
    "dn|dadra and nagar haveli|dadra nagar haveli|daman and diu|dd|dnhdd|"
    "dadra and nagar haveli and daman and diu",
    "dl|delhi|new delhi|nct of delhi", "jk|jammu and kashmir|jammu kashmir", "la|ladakh",
    "ld|lakshadweep", "py|puducherry|pondicherry",
]


def skel_word(w):
    """Vowel-less 'skeleton' of one word: limited / limittedd -> lmtd; ph->f, ck->k, 0->o, 1->l."""
    if w.isdigit():
        return w
    x = w.replace("ph", "f").replace("ck", "k").replace("0", "o").replace("1", "l")
    x = DBL.sub(r"\1", x)
    y = DBL.sub(r"\1", VOW.sub("", x))
    return y or x


def skel_key(text):
    return "".join(skel_word(w) for w in text.split())


def _build_state_maps():
    maps = {}
    us_exact, us_skel = {}, {}
    for item in US_STATES.split(";"):
        code, name = item[:2], item[3:]
        c = code.lower()
        us_exact[c] = c
        us_exact[name] = c
        if len(skel_key(name)) >= 3:
            us_skel.setdefault(skel_key(name), c)
    ind_exact, ind_skel = {}, {}
    for item in INDIA_STATES:
        keys = item.split("|")
        c = keys[0]
        for k in keys:
            ind_exact[k] = c
            if len(k) > 2 and len(skel_key(k)) >= 3:
                ind_skel.setdefault(skel_key(k), c)
    for k in ("us", "usa", "united states", "united states of america"):
        maps[k] = (us_exact, us_skel)
    maps["india"] = (ind_exact, ind_skel)
    return maps


STATE_MAPS = _build_state_maps()


@functools.lru_cache(maxsize=500_000)
def _skel_state_lookup(ckey, key):
    """Fuzzy match of a transliterated (non-Latin origin) state name against the skeleton list."""
    exact, skel = STATE_MAPS[ckey]
    k = skel_key(key)
    if len(k) < 3:
        return ""
    hit = skel.get(k)
    if hit:
        return hit
    best = process.extractOne(k, list(skel.keys()), scorer=fuzz.ratio, score_cutoff=78)
    return skel[best[0]] if best else ""


def _asc(s):
    return (s if s.isascii() else unidecode(s)).lower()


def clean_name(raw):
    s = _asc(raw).replace("'", "").replace("`", "")
    s = DOT_ACR.sub("", s)
    s = WEB_RE.sub(" ", s)
    s = s.replace("&", " and ")
    toks = [t for t in NONALNUM.sub(" ", s).split() if t not in TITLES]
    nf = " ".join(LEGAL_MAP.get(t, t) for t in toks)
    core = [t for t in toks if t not in LEGAL] or toks
    nm = " ".join(core)
    sk = " ".join(skel_word(t) for t in core)
    return nm, nf, sk, nm.replace(" ", "")


def clean_addr(raw, ckey):
    raw_nonascii = not raw.isascii()
    s = JUNK.sub(" ", _asc(raw))
    parts = [NONALNUM.sub(" ", p).strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    st = ""
    maps = STATE_MAPS.get(ckey)
    if maps and parts:
        exact = maps[0]
        n = len(parts)
        order = [n - 1, 0] + ([n - 2] if n >= 3 else [])
        for idx in order:
            key = TRAIL_DIG.sub("", parts[idx]).strip()
            v = exact.get(key)
            if v is None and raw_nonascii and key:
                v = _skel_state_lookup(ckey, key)
            if v:
                st = v
                del parts[idx]
                break
    toks = " ".join(parts).split()
    out = []
    for i, t in enumerate(toks):
        if t == "r" and i + 1 < len(toks) and toks[i + 1] in R_RULE_NEXT:
            out.append("rue")
        else:
            out.append(ADDR_ABBR.get(t, t))
    hn = zp = ""
    nums = []
    for t in out:
        if any(ch.isdigit() for ch in t):
            nums.append(t)
            if t.isdigit() and len(t) in (5, 6):
                if not zp:
                    zp = t
            elif not hn:
                hn = t
    return " ".join(out), st, hn, zp, " ".join(nums)


def clean_chunk(args):
    names, addrs, ckeys = args
    cols = [[] for _ in range(10)]
    for nm_raw, ad_raw, ck in zip(names, addrs, ckeys):
        nm, nf, sk, nn = clean_name(nm_raw)
        ad, st, hn, zp, nums = clean_addr(ad_raw, ck)
        for c, v in zip(cols, (nm, nf, sk, nn, ad, st, hn, zp, nums, 0 if nm_raw.isascii() else 1)):
            c.append(v)
    return cols


class Recs:
    """Cleaned records of one country/source group (parallel python lists)."""
    FIELDS = ("nm", "nf", "sk", "nn", "ad", "st", "hn", "zp", "nums", "nl")

    def __init__(self, df, nproc):
        self.ids = df["entity_id"].tolist()
        self.s3 = [1 if i.startswith("S3") else 0 for i in self.ids]
        names = df["business_name"].tolist()
        addrs = df["business_address"].tolist()
        ckeys = df["country"].str.strip().str.lower().tolist()
        step = 20_000
        chunks = [(names[i:i + step], addrs[i:i + step], ckeys[i:i + step]) for i in range(0, len(names), step)]
        cols = [[] for _ in self.FIELDS]
        ctx = get_ctx()
        if ctx is not None and nproc > 1 and len(chunks) > 1:
            with ctx.Pool(nproc) as pool:
                for res in pool.imap(clean_chunk, chunks):
                    for c, r in zip(cols, res):
                        c.extend(r)
        else:
            for ch in chunks:
                for c, r in zip(cols, clean_chunk(ch)):
                    c.extend(r)
        for f, c in zip(self.FIELDS, cols):
            setattr(self, f, c)
        self.ids_arr = np.array(self.ids, dtype=object)


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------
def find_file(name, data_dir=None):
    roots = [r for r in [data_dir, "/kaggle/input", os.getcwd()] if r]
    for r in roots:
        hits = glob.glob(os.path.join(r, "**", name), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {roots}")


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8-sig", encoding_errors="replace")


def read_source(path):
    df = read_tsv(path)
    df.columns = ["entity_id", "business_name", "business_address", "country"]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


# ----------------------------------------------------------------------------------------------
# TF-IDF (hashed) + shortlist
# ----------------------------------------------------------------------------------------------
_HV = None


def _hash_chunk(texts):
    return _HV.transform(texts)


def hash_matrix(texts, hv, nproc):
    global _HV
    _HV = hv
    step = 100_000
    chunks = [texts[i:i + step] for i in range(0, len(texts), step)]
    ctx = get_ctx()
    if ctx is not None and nproc > 1 and len(chunks) > 1:
        with ctx.Pool(nproc) as pool:
            mats = list(pool.imap(_hash_chunk, chunks))
    else:
        mats = [_hash_chunk(c) for c in chunks]
    return sp.vstack(mats, format="csr")


def tfidf_pair(q_texts, p_texts, hv, max_df_frac, nproc):
    """Hash both lists together, weight by idf (drop very common / singleton features), l2-normalise."""
    X = hash_matrix(list(q_texts) + list(p_texts), hv, nproc)
    n = X.shape[0]
    df = np.bincount(X.indices, minlength=X.shape[1])
    cap = max(20, int(max_df_frac * n))
    idf = (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
    idf[(df > cap) | (df < 2)] = 0.0
    X.data = (1.0 + np.log(X.data)).astype(np.float32)
    X = (X @ sp.diags(idf)).tocsr()
    X.eliminate_zeros()
    X = normalize(X, copy=False).astype(np.float32)
    nq = len(q_texts)
    return X[:nq].tocsr(), X[nq:].tocsr()


def make_hashers():
    hn = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), lowercase=False, n_features=CFG["HASH_FEATS"],
                           alternate_sign=False, norm=None, dtype=np.float32)
    ha = HashingVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), lowercase=False,
                           n_features=CFG["HASH_FEATS"], alternate_sign=False, norm=None, dtype=np.float32)
    return hn, ha


def topk_csr(M, k):
    """Row/col ids of the k largest entries of every row of a sparse matrix."""
    M = M.tocsr()
    ip, ix, dt = M.indptr, M.indices, M.data
    rows, cols = [], []
    for r in range(M.shape[0]):
        a, b = ip[r], ip[r + 1]
        if b == a:
            continue
        if b - a > k:
            sel = np.argpartition(dt[a:b], (b - a) - k)[(b - a) - k:] + a
            c = ix[sel]
        else:
            c = ix[a:b]
        rows.append(np.full(len(c), r, np.int64))
        cols.append(c.astype(np.int64))
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(rows), np.concatenate(cols)


def row_dot(A, B, qi, pj):
    return np.asarray(A[qi].multiply(B[pj]).sum(axis=1)).ravel().astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Pair features
# ----------------------------------------------------------------------------------------------
STR_FEATS = ["nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "nm_jw", "sk_ratio", "sk_tset", "sk_jw",
             "sk_partial", "nn_ratio", "nn_partial", "nf_tset", "ad_tset", "ad_tsort", "ad_partial", "ad_jw",
             "ad_ratio", "hn_ratio", "nums_jacc", "zp_match", "st_match", "q_nlen", "c_nlen", "nlen_diff",
             "q_alen", "c_alen", "c_nonlatin", "c_is_s3", "first_tok_eq", "sk_first_eq", "sk_tok_jacc"]
GRAPH_FEATS = ["name_sim", "addr_sim", "sim_sum", "rank_s", "rank_n", "rank_a", "gap_best", "rel_best",
               "gap2", "n_cands", "gap_n", "gap_a"]
FEATURES = STR_FEATS + GRAPH_FEATS
NS = len(STR_FEATS)

_G = {}


def _feat_worker(args):
    qi, pj = args
    Q, P = _G["Q"], _G["P"]
    qnm, qnf, qsk, qnn, qad, qst, qhn, qzp, qnums = Q.nm, Q.nf, Q.sk, Q.nn, Q.ad, Q.st, Q.hn, Q.zp, Q.nums
    pnm, pnf, psk, pnn, pad, pst, phn, pzp, pnums = P.nm, P.nf, P.sk, P.nn, P.ad, P.st, P.hn, P.zp, P.nums
    pnl, ps3 = P.nl, P.s3
    ratio, tset, tsort, part, jw = fuzz.ratio, fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_ratio, \
        JaroWinkler.similarity
    qi = qi.tolist()
    pj = pj.tolist()
    out = np.zeros((len(qi), NS), np.float32)
    for k in range(len(qi)):
        i, j = qi[k], pj[k]
        r = out[k]
        a, b = qnm[i], pnm[j]
        if a and b:
            r[0] = ratio(a, b)
            r[1] = tset(a, b)
            r[2] = tsort(a, b)
            r[3] = part(a, b)
            r[4] = jw(a, b) * 100
            r[28] = 1.0 if a.split(" ", 1)[0] == b.split(" ", 1)[0] else 0.0
        sa, sb = qsk[i], psk[j]
        if sa and sb:
            r[5] = ratio(sa, sb)
            r[6] = tset(sa, sb)
            r[7] = jw(sa, sb) * 100
            r[8] = part(sa, sb)
            r[29] = 1.0 if sa[0] == sb[0] else 0.0
            ta, tb = set(sa.split()), set(sb.split())
            r[30] = len(ta & tb) / max(1, len(ta | tb))
        na, nb = qnn[i], pnn[j]
        if na and nb:
            r[9] = ratio(na, nb)
            r[10] = part(na, nb)
        fa, fb = qnf[i], pnf[j]
        if fa and fb:
            r[11] = tset(fa, fb)
        da, db = qad[i], pad[j]
        if da and db:
            r[12] = tset(da, db)
            r[13] = tsort(da, db)
            r[14] = part(da, db)
            r[15] = jw(da, db) * 100
            r[16] = ratio(da, db)
        h1, h2 = qhn[i], phn[j]
        r[17] = ratio(h1, h2) / 100.0 if (h1 and h2) else -1.0
        n1, n2 = qnums[i], pnums[j]
        if n1 and n2:
            s1, s2 = set(n1.split()), set(n2.split())
            r[18] = len(s1 & s2) / max(1, len(s1 | s2))
        else:
            r[18] = -1.0
        z1, z2 = qzp[i], pzp[j]
        r[19] = (1.0 if z1 == z2 else 0.0) if (z1 and z2) else -1.0
        t1, t2 = qst[i], pst[j]
        r[20] = (1.0 if t1 == t2 else 0.0) if (t1 and t2) else -1.0
        r[21] = len(a)
        r[22] = len(b)
        r[23] = abs(len(a) - len(b))
        r[24] = len(da)
        r[25] = len(db)
        r[26] = pnl[j]
        r[27] = ps3[j]
    return out


class FeatRunner:
    def __init__(self, Q, P, nproc):
        _G["Q"], _G["P"] = Q, P
        self.nproc = nproc
        ctx = get_ctx()
        self.pool = ctx.Pool(nproc) if (ctx is not None and nproc > 1) else None

    def run(self, qg, pg):
        if self.pool is None or len(qg) < 2000:
            return _feat_worker((qg, pg))
        parts = min(len(qg), self.nproc * 3)
        qs, ps = np.array_split(qg, parts), np.array_split(pg, parts)
        return np.vstack(self.pool.map(_feat_worker, list(zip(qs, ps))))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def graph_feats(qg, ns, as_):
    s = ns + as_
    df = pd.DataFrame({"q": qg, "n": ns, "a": as_, "s": s})
    g = df.groupby("q", sort=False)
    rs = g["s"].rank(ascending=False, method="first").values
    rn = g["n"].rank(ascending=False, method="min").values
    ra = g["a"].rank(ascending=False, method="min").values
    best = g["s"].transform("max").values
    cnt = g["s"].transform("size").values
    bn = g["n"].transform("max").values
    ba = g["a"].transform("max").values
    df["rs"] = rs
    sec = df[df["rs"] == 2].set_index("q")["s"]
    sec_v = df["q"].map(sec).fillna(0).values
    cols = [ns, as_, s, rs, rn, ra, s - best, s / (best + 1e-6), best - sec_v, cnt, ns - bn, as_ - ba]
    return np.column_stack(cols).astype(np.float32)


def run_country(Q, P, sink, nproc):
    """Shortlist + features for one country. Calls sink(q_idx, pool_idx, X) per chunk of S1 records."""
    nQ, nP = len(Q.ids), len(P.ids)
    if nQ == 0 or nP == 0:
        return
    hn, ha = make_hashers()
    t = time.time()
    Qn, Pn = tfidf_pair(Q.sk, P.sk, hn, CFG["NAME_MAX_DF"], nproc)
    Qa, Pa = tfidf_pair(Q.ad, P.ad, ha, CFG["ADDR_MAX_DF"], nproc)
    log(f"  tf-idf ready ({time.time() - t:.0f}s)  S1={nQ}  pool={nP}")

    codes, uniq = pd.factorize(pd.Series(Q.st + P.st))
    empty = int(np.flatnonzero(uniq.values == "")[0]) if (uniq.values == "").any() else -1
    qs, ps = codes[:nQ], codes[nQ:]
    counts = pd.Series(qs).value_counts()
    runner = FeatRunner(Q, P, nproc)
    done = 0
    t = time.time()
    last_log = t
    try:
        for code in counts.index:
            qsel = np.flatnonzero(qs == code)
            if code == empty:
                pidx = np.arange(nP)
            else:
                pidx = np.flatnonzero((ps == code) | (ps == empty))
            if len(pidx) == 0:
                continue
            nB = len(pidx)
            PnT = Pn[pidx].T.tocsr()
            PaT = Pa[pidx].T.tocsr()
            step = 1500 if nB < 100_000 else (500 if nB < 400_000 else 250)
            for a in range(0, len(qsel), step):
                ql = qsel[a:a + step]
                r1, c1 = topk_csr(Qn[ql] @ PnT, CFG["K_NAME"])
                r2, c2 = topk_csr(Qa[ql] @ PaT, CFG["K_ADDR"])
                key = np.unique(np.concatenate([r1, r2]) * nB + np.concatenate([c1, c2]))
                if len(key) == 0:
                    done += len(ql)
                    continue
                qg, pg = ql[key // nB], pidx[key % nB]
                ns, as_ = row_dot(Qn, Pn, qg, pg), row_dot(Qa, Pa, qg, pg)
                X = np.hstack([runner.run(qg, pg), graph_feats(qg, ns, as_)])
                sink(qg, pg, X)
                done += len(ql)
            if time.time() - last_log > 60:
                last_log = time.time()
                log(f"  {done}/{nQ} S1 done  ({time.time() - t:.0f}s)")
    finally:
        runner.close()
    log(f"  country shortlist+features done ({time.time() - t:.0f}s)")


# ----------------------------------------------------------------------------------------------
# Decision rule + metric
# ----------------------------------------------------------------------------------------------
def decide_idx(q, c, p, thr, rel):
    """Indices (into q/c/p) of the pairs kept: p>=thr, each candidate only to its best S1, then
    only candidates within `rel` x the S1's best remaining score."""
    m = np.flatnonzero(p >= thr)
    if len(m) == 0:
        return m
    order = m[np.argsort(-p[m], kind="stable")]
    _, first = np.unique(c[order], return_index=True)
    keep = order[np.sort(first)]
    if rel > 0 and len(keep):
        mx = pd.Series(p[keep]).groupby(q[keep]).transform("max").values
        keep = keep[p[keep] >= rel * mx]
    return keep


def macro_f05(q, y, sel, ntrue, eval_q):
    n = len(ntrue)
    qs = q[sel]
    npred = np.bincount(qs, minlength=n).astype(np.float64)
    tp = np.bincount(qs, weights=y[sel], minlength=n)
    P = tp / np.maximum(npred, 1)
    R = tp / np.maximum(ntrue, 1)
    F = np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)
    F = np.where((npred == 0) & (ntrue == 0), 1.0, F)
    return float(F[eval_q].mean())




# ----------------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------------
def train_stage(a):
    nproc = a.nproc
    n_train, n_val = (4000, 1500) if a.quick else (CFG["N_TRAIN"], CFG["N_VAL"])
    log("loading training files")
    s1 = read_source(find_file("train_source1.tsv", a.data_dir))
    s2 = read_source(find_file("train_source2.tsv", a.data_dir))
    s3 = read_source(find_file("train_source3.tsv", a.data_dir))
    gt = read_tsv(find_file("train_ground_truth.tsv", a.data_dir))
    gt.columns = ["s1", "matches"]
    rng = np.random.RandomState(CFG["SEED"])
    pick = rng.permutation(len(s1))[:n_train + n_val]
    samp = s1.iloc[pick].reset_index(drop=True)
    is_val = np.arange(len(samp)) < n_val
    gtd = dict(zip(gt["s1"], gt["matches"]))
    gsets = [set(x for x in gtd.get(i, "").split(",") if x) for i in samp["entity_id"]]
    ntrue = np.array([len(g) for g in gsets], dtype=np.int64)
    log(f"S1 sample: {len(samp)} (val {n_val}); share with no match: {(ntrue == 0).mean():.3f}")
    del s1, gt, gtd

    Xs, ys, qs_all, cs_all = [], [], [], []
    offset = 0
    for ctry in sorted(samp["country"].unique()):
        rows = np.flatnonzero(samp["country"].values == ctry)
        pool_df = pd.concat([s2[s2["country"] == ctry], s3[s3["country"] == ctry]], ignore_index=True)
        if a.quick:
            need = set().union(*[gsets[i] for i in rows])
            keep = pool_df["entity_id"].isin(need).values | (rng.rand(len(pool_df)) < 0.05)
            pool_df = pool_df[keep].reset_index(drop=True)
        log(f"country {ctry}: S1 {len(rows)}, pool {len(pool_df)} -> cleaning")
        Q = Recs(samp.iloc[rows].reset_index(drop=True), nproc)
        P = Recs(pool_df, nproc)
        n_pool = len(P.ids)
        del pool_df
        gs = [gsets[i] for i in rows]

        def sink(qg, pg, X, rows=rows, gs=gs, P=P, offset=offset):
            y = np.fromiter((P.ids[j] in gs[i] for i, j in zip(qg.tolist(), pg.tolist())),
                            dtype=np.int8, count=len(qg))
            Xs.append(X)
            ys.append(y)
            qs_all.append(rows[qg])
            cs_all.append(pg + offset)          # candidate id, unique across countries

        run_country(Q, P, sink, nproc)
        offset += n_pool
        del Q, P
    del s2, s3

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    q = np.concatenate(qs_all)
    c = np.concatenate(cs_all)
    del Xs, ys, qs_all, cs_all
    vm = is_val[q]
    eval_q = np.flatnonzero(is_val)
    log(f"pairs: {len(y)}  positives: {int(y.sum())}  shortlist recall: {y.sum() / max(1, ntrue.sum()):.4f}")
    perfect = macro_f05(q, y.astype(np.float64), np.flatnonzero(vm & (y == 1)), ntrue, eval_q)
    log(f"perfect-model upper bound on validation: {perfect:.4f}")
    log(f"'predict nothing' baseline on validation: "
        f"{macro_f05(q, y.astype(np.float64), np.zeros(0, np.int64), ntrue, eval_q):.4f}")

    tr = ~vm
    params = dict(objective="binary", metric="binary_logloss", learning_rate=0.08, num_leaves=255,
                  min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=nproc, verbose=-1, seed=CFG["SEED"])
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=FEATURES, free_raw_data=False)
    dva = lgb.Dataset(X[vm], y[vm], reference=dtr)
    log("training LightGBM")
    bst = lgb.train(params, dtr, num_boost_round=150 if a.quick else CFG["NUM_ROUNDS"], valid_sets=[dva],
                    callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)])
    vidx = np.flatnonzero(vm)
    pv = bst.predict(X[vidx], num_iteration=bst.best_iteration).astype(np.float32)
    qv, cv, yv = q[vidx], c[vidx], y[vidx].astype(np.float64)

    log("tuning decision rule on held-out S1")
    best = (-1.0, 0.65, 0.0)
    for rel in (0.0, 0.5, 0.7, 0.85):
        for thr in np.arange(0.30, 0.96, 0.05):
            sel = decide_idx(qv, cv, pv, thr, rel)
            f = macro_f05(qv, yv, sel, ntrue, eval_q)
            if f > best[0]:
                best = (f, float(thr), rel)
        log(f"  rel={rel}: best so far F0.5={best[0]:.4f} at thr={best[1]:.2f}, rel={best[2]}")
    log(f"VALIDATION F0.5 = {best[0]:.4f}  (threshold {best[1]:.2f}, rel {best[2]})")

    os.makedirs(a.model_dir, exist_ok=True)
    bst.save_model(os.path.join(a.model_dir, "model.txt"), num_iteration=bst.best_iteration)
    imp = sorted(zip(FEATURES, bst.feature_importance("gain")), key=lambda t: -t[1])
    with open(os.path.join(a.model_dir, "model_meta.json"), "w") as f:
        json.dump(dict(thr=best[1], rel=best[2], val_f05=best[0], features=FEATURES,
                       best_iteration=bst.best_iteration, cfg=CFG,
                       importance=[(k, float(v)) for k, v in imp]), f, indent=1)
    log("top features: " + ", ".join(k for k, _ in imp[:8]))
    log(f"model saved to {a.model_dir}")


# ----------------------------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------------------------
def _lines(ids_q, qarr, cand_ids, carr):
    """'S1<TAB>c1,c2,...' lines for every S1 (empty list if it has no pair)."""
    if len(qarr):
        s = pd.DataFrame({"q": qarr, "c": cand_ids[carr]}).groupby("q")["c"].agg(",".join)
        full = s.reindex(range(len(ids_q))).fillna("").values
    else:
        full = [""] * len(ids_q)
    return [f"{i}\t{v}" for i, v in zip(ids_q, full)]


def _tag(country):
    return re.sub(r"\W+", "_", country)


def predict_stage(a):
    nproc = a.nproc
    with open(os.path.join(a.model_dir, "model_meta.json")) as f:
        meta = json.load(f)
    assert meta["features"] == FEATURES, "feature list changed since training - retrain"
    bst = lgb.Booster(model_file=os.path.join(a.model_dir, "model.txt"))
    thr, rel = meta["thr"], meta["rel"]
    log(f"model loaded; decision rule thr={thr:.2f} rel={rel}  (val F0.5 {meta['val_f05']:.4f})")

    p1 = find_file("test_source1.tsv", a.data_dir)
    log("loading test files")
    s1 = read_source(p1)
    s2 = read_source(find_file("test_source2.tsv", a.data_dir))
    s3 = read_source(find_file("test_source3.tsv", a.data_dir))
    log(f"test S1={len(s1)}  S2={len(s2)}  S3={len(s3)}")
    out_dir = os.path.join(a.out_dir, "output_quick" if a.quick else "output")
    parts = os.path.join(out_dir, "parts")
    os.makedirs(parts, exist_ok=True)
    rng = np.random.RandomState(1)

    countries = sorted(s1["country"].unique())
    for ctry in countries:
        fm = os.path.join(parts, "m_" + _tag(ctry) + ".tsv")
        fc = os.path.join(parts, "c_" + _tag(ctry) + ".tsv")
        if os.path.exists(fm) and os.path.exists(fc) and not a.quick:
            log(f"country {ctry}: parts already exist, skipping")
            continue
        q_df = s1[s1["country"] == ctry].reset_index(drop=True)
        pool_df = pd.concat([s2[s2["country"] == ctry], s3[s3["country"] == ctry]], ignore_index=True)
        if a.quick:
            q_df = q_df.head(1500)
            pool_df = pool_df.iloc[rng.permutation(len(pool_df))[:200_000]].reset_index(drop=True)
        log(f"country {ctry}: S1 {len(q_df)}, pool {len(pool_df)} -> cleaning")
        Q, P = Recs(q_df, nproc), Recs(pool_df, nproc)
        del pool_df
        qs, cs, ps = [], [], []

        def sink(qg, pg, X):
            qs.append(qg.astype(np.int32))
            cs.append(pg.astype(np.int32))
            ps.append(bst.predict(X, num_threads=nproc).astype(np.float32))

        run_country(Q, P, sink, nproc)
        if qs:
            q, c, p = np.concatenate(qs), np.concatenate(cs), np.concatenate(ps)
        else:
            q = c = np.zeros(0, np.int64)
            p = np.zeros(0, np.float32)
        sel = decide_idx(q, c, p, thr, rel)
        log(f"country {ctry}: {len(q)} candidate pairs, {len(sel)} kept "
            f"({len(sel) / max(1, len(Q.ids)):.2f} per S1, S1 with >=1 match: "
            f"{len(np.unique(q[sel])) / max(1, len(Q.ids)):.3f})")
        with open(fm, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(_lines(Q.ids, q[sel], P.ids_arr, c[sel])) + "\n")
        with open(fc, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(_lines(Q.ids, q, P.ids_arr, c)) + "\n")
        del Q, P, q, c, p, qs, cs, ps

    for name, prefix, head in (("matching_results.tsv", "m_", "source1_entity_id\tmatched_entity_ids"),
                               ("candidate_pairs.tsv", "c_", "source1_entity_id\tcandidate_entity_ids")):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="") as fo:
            fo.write(head + "\n")
            for ctry in countries:
                with open(os.path.join(parts, prefix + _tag(ctry) + ".tsv"), encoding="utf-8") as fi:
                    fo.write(fi.read())
    log(f"wrote {out_dir}/matching_results.tsv and candidate_pairs.tsv")

    if not a.quick:
        try:
            val = find_file("validate_submission.py", a.data_dir)
            log("running official validator")
            subprocess.call([sys.executable, val, "--matching", os.path.join(out_dir, "matching_results.tsv"),
                             "--candidate", os.path.join(out_dir, "candidate_pairs.tsv"),
                             "--test-dir", os.path.dirname(p1)])
        except FileNotFoundError:
            log("validate_submission.py not found - upload it or run it manually before submitting")


def main(argv=None):
    """argv=None reads the command line; in a notebook call e.g. main(["--stage", "all", "--quick"])."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "predict", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (small sample, output_quick/)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default="/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
    ap.add_argument("--nproc", type=int, default=os.cpu_count() or 4)
    a, _unknown = ap.parse_known_args(argv)      # ignores the extra arguments Jupyter/Colab add
    a.model_dir = os.path.join(a.out_dir, "model_quick" if a.quick else "model")
    log(f"stage={a.stage} quick={a.quick} nproc={a.nproc} out={a.out_dir}")
    if a.stage in ("train", "all"):
        train_stage(a)
    if a.stage in ("predict", "all"):
        predict_stage(a)
    log("done")


def _in_notebook():
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


if __name__ == "__main__":
    if _in_notebook():
        print("Loaded. In a NEW cell run:  main(['--stage', 'all', '--quick'])   (smoke test)\n"
              "then, for the real run:     main(['--stage', 'all'])")
    else:
        main()
