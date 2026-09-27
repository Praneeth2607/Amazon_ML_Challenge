"""
Business Entity Resolution - Amazon ML Challenge 2026 - v10 (single file for Kaggle).
A different method from v9: TWO STAGES.
  Stage 1 scores every (business, candidate) pair with the strong v8 features + v7 exact checks (house number, legal form).
  Stage 2 looks at ALL candidates of a business together (Source 2 + Source 3) and asks "does this candidate look like
  the business's other strong candidates?" (v2's idea, +0.004 on v1, here on a much stronger stage 1).
  Then a team of 3 models decides; cut-offs are tuned per country; a France check is printed.

TWO SESSIONS (each must finish inside Kaggle's 12 hours)
  SESSION 1 - training (about 4-6 h):   RUN = "train"                    <- the default below
  SESSION 2 - prediction (about 6-9 h): RUN = "predict,combine,check"    <- change the RUN line, and use
      Add Input -> Notebook Output to attach the finished session-1 version (the models are copied from it).
  If prediction does not fit in one session, split it by country and run two sessions, e.g.
      ONLY_COUNTRIES = "France,US"   in one, and   "India"   in the other (attach the first one's output in the second).
      The last session runs  RUN = "predict,combine,check"  with all earlier outputs attached; batches already done are skipped.
  Only start session 2 if session 1 ended with a good "BEST val F0.5 overall" (v8 was 0.9673, v9 is being run separately).
  SMOKE TEST (optional, about 30 min, tiny data): set SMOKE = "1", run, then set it back to "0".
  After a full run, FRANCE_MARGIN can be changed and RUN = "combine" re-run alone (takes minutes).
Paste this whole file into ONE notebook code cell (or save it and run: !python er_v10_kaggle.py).
No external data or APIs are used, only the competition files.
"""
import os, sys, subprocess
# ===================== EDIT ONLY THESE LINES =====================
RUN = "train"          # "train"  |  "predict,combine,check"  |  "combine"  |  "train,predict,combine,check"
SMOKE = "0"            # "1" = tiny smoke test
ONLY_COUNTRIES = ""    # "" = all countries; or e.g. "France,US" to predict only those in this session
# =================================================================
os.environ["STAGES"] = RUN
os.environ["SMOKE"] = SMOKE
os.environ["ONLY_COUNTRIES"] = ONLY_COUNTRIES
STAGES = RUN.split(",")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + ['rapidfuzz', 'sparse_dot_topn', 'unidecode', 'polars', 'lightgbm', 'xgboost', 'catboost', 'jellyfish'])


# ======================================================================
# 1. v1 pipeline: loading, cleaning, shortlist, stage-1 features
# ======================================================================
# =====================================================================
# Business Entity Resolution: full pipeline (clean -> shortlist -> features -> LightGBM -> decide)
# No external data or services are used anywhere. Only the provided files.
# =====================================================================
import os, re, gc, glob, time, json
import numpy as np
import pandas as pd
import polars as pl
from multiprocessing import Pool
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
import lightgbm as lgb

N_CPU = os.cpu_count() or 4

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------------------------------------------------------------
# 1. Loading. Files are tab-separated, all columns read as text.
#    Handles files split into *_part_N.tsv (BOM / CRLF / repeated header).
# ---------------------------------------------------------------------
def find_file(root, name):
    hits = sorted(glob.glob(f"{root}/**/{name}.tsv", recursive=True))
    if hits:
        return [hits[0]]
    parts = glob.glob(f"{root}/**/{name}_part_*.tsv", recursive=True)
    parts.sort(key=lambda p: int(re.findall(r"_part_(\d+)", p)[-1]))
    if not parts:
        raise FileNotFoundError(f"{name}.tsv not found under {root}")
    return parts

def load(root, name):
    frames, header = [], None
    for p in find_file(root, name):
        txt = open(p, encoding="utf-8-sig").read().replace("\r\n", "\n").replace("\r", "")
        if txt.startswith(("entity_id\t", "source1_entity_id\t")):
            header, txt = txt.split("\n", 1)
        frames.append(txt)
    body = "".join(f if f.endswith("\n") else f + "\n" for f in frames)
    df = pl.read_csv((header + "\n" + body).encode(), separator="\t", quote_char=None, infer_schema_length=0)
    return df.fill_null("").to_pandas()

# ---------------------------------------------------------------------
# 2. Cleaning (normalisation). Hand-written rules only.
# ---------------------------------------------------------------------
ABBR = {"rd":"road","st":"street","ave":"avenue","av":"avenue","blvd":"boulevard","bd":"boulevard","hwy":"highway",
        "dr":"drive","ln":"lane","ct":"court","pl":"place","pkwy":"parkway","cir":"circle","ste":"suite","fl":"floor",
        "apt":"apartment","no":"number","nr":"near","opp":"opposite","bldg":"building","ind":"industrial",
        "dist":"district","r":"rue","ch":"chemin","n":"north","s":"south","e":"east","w":"west","mt":"mount",
        "ft":"fort","hts":"heights","sq":"square","imp":"impasse","all":"allee"}
NULLS = {"null", "na", "none", "nan"}
# legal words, written in "skeleton" form (see skel_tok): covers Limited/Ltd/limittedd, Private/Pvt/praaivett, LLC, SARL ...
LEGAL_SK = {"lmtd","ltd","prwt","pwt","llc","llp","inc","ncrprtd","crp","crprtn","c","cmpny","plc","srl","ss","sa",
            "sc","erl","snc","gmbh","t","m","s","d","dba","lp","pc","pllc","pa","ms","and"}

def base(s):
    s = unidecode(str(s)).lower()                                   # also turns Hindi/Telugu/... script into Latin letters
    s = re.sub(r"<?null>?|n/a", " ", s)
    s = re.sub(r"https?://|www\.|\.(com|net|org|in|co|io|biz|fr|us)\b", " ", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def skel_tok(t):
    """Spelling-robust 'skeleton' of a word: drop vowels, merge ph/f, bh/b ..., collapse double letters.
    'limited' and the transliterated 'limittedd' both become 'lmtd'."""
    if not t or t.isdigit():
        return t
    t = t.replace("0", "o").replace("1", "l") if re.search(r"[a-z]", t) else t   # leetspeak: C1ean -> Clean
    t = re.sub(r"ph", "f", t)
    t = re.sub(r"([bkgtdcs])h", r"\1", t)
    t = t.replace("v", "w").replace("q", "k").replace("z", "s").replace("x", "ks")
    t = t[0] + re.sub(r"[aeiouy]", "", t[1:])
    return re.sub(r"(.)\1+", r"\1", t)

US = {"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co","connecticut":"ct",
"delaware":"de","florida":"fl","georgia":"ga","hawaii":"hi","idaho":"id","illinois":"il","indiana":"in","iowa":"ia",
"kansas":"ks","kentucky":"ky","louisiana":"la","maine":"me","maryland":"md","massachusetts":"ma","michigan":"mi",
"minnesota":"mn","mississippi":"ms","missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv","new hampshire":"nh",
"new jersey":"nj","new mexico":"nm","new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok",
"oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc","south dakota":"sd","tennessee":"tn",
"texas":"tx","utah":"ut","vermont":"vt","virginia":"va","washington":"wa","west virginia":"wv","wisconsin":"wi",
"wyoming":"wy","district of columbia":"dc","puerto rico":"pr"}
IN = {"andhra pradesh":"ap","arunachal pradesh":"ar","assam":"as","bihar":"br","chhattisgarh":"cg","goa":"ga",
"gujarat":"gj","haryana":"hr","himachal pradesh":"hp","jharkhand":"jh","karnataka":"ka","kerala":"kl",
"madhya pradesh":"mp","maharashtra":"mh","manipur":"mn","meghalaya":"ml","mizoram":"mz","nagaland":"nl","odisha":"od",
"orissa":"od","punjab":"pb","rajasthan":"rj","sikkim":"sk","tamil nadu":"tn","telangana":"ts","tripura":"tr",
"uttar pradesh":"up","uttarakhand":"uk","uttaranchal":"uk","west bengal":"wb","delhi":"dl","new delhi":"dl",
"jammu and kashmir":"jk","ladakh":"la","puducherry":"py","pondicherry":"py","chandigarh":"ch",
"andaman and nicobar islands":"an","lakshadweep":"ld"}
MERGE_ST = {"ts": "ap"}            # Telangana was split from Andhra Pradesh; the data mixes them
_STATE = {}
for _d in (US, IN):
    for _full, _code in _d.items():
        _STATE[tuple(skel_tok(t) for t in _full.split())] = _code
_CODES = set(US.values()) | set(IN.values())

def states_of(raw_addr):
    """All state/province codes mentioned in an address ('' when none). Country-agnostic: unknown countries get ''."""
    found = set()
    for seg in unidecode(str(raw_addr)).lower().split(","):
        toks = re.sub(r"[^a-z ]", " ", seg).split()
        if not toks:
            continue
        if len(toks) == 1 and toks[0] in _CODES:
            found.add(toks[0]); continue
        sk = [skel_tok(t) for t in toks]
        for n in (4, 3, 2, 1):
            for i in range(len(sk) - n + 1):
                key = tuple(sk[i:i + n])
                if key in _STATE and (n > 1 or len(key[0]) >= 3):
                    found.add(_STATE[key])
    return "|".join(sorted({MERGE_ST.get(x, x) for x in found}))

def _clean_rows(rows):
    out = []
    for name, addr in rows:
        nb = base(name)
        sk = [skel_tok(t) for t in nb.split()]
        core = [t for t in sk if t not in LEGAL_SK] or sk
        an = " ".join(ABBR.get(t, t) for t in base(addr).split() if t not in NULLS)
        out.append((nb, " ".join(core), "".join(core), an,
                    " ".join(skel_tok(t) for t in an.split()), states_of(addr)))
    return out

def prepare(df):
    """Adds the cleaned columns. Runs on all CPU cores."""
    rows = list(zip(df.business_name.tolist(), df.business_address.tolist()))
    step = 50_000
    with Pool(N_CPU) as pool:
        res = pool.map(_clean_rows, [rows[i:i + step] for i in range(0, len(rows), step)])
    res = [r for chunk in res for r in chunk]
    cols = ["name_b", "name_sk", "name_skc", "addr_n", "addr_sk", "states"]
    for j, c in enumerate(cols):
        df[c] = [r[j] for r in res]
    df.drop(columns=["business_address"], inplace=True)
    df["nonascii"] = df.business_name.str.contains(r"[^\x00-\x7f]", regex=True).astype(np.int8)
    df.drop(columns=["business_name"], inplace=True)
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------
# 3. Shortlist ("blocking"). For each Source 1 record, look only inside the same country and
#    the same state, and keep the top-k most similar names and the top-k most similar addresses.
#    Records whose state could not be read form a separate pool per country.
# ---------------------------------------------------------------------
def _vec_name():
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 3), dtype=np.float32, sublinear_tf=True)
def _vec_addr():
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), dtype=np.float32, sublinear_tf=True)

class Index:
    """TF-IDF indexes over one pool of candidate records (one country of Source 2 or Source 3)."""
    def __init__(self, B, k_state=15, k_unk=10):
        self.B, self.k_state, self.k_unk = B, k_state, k_unk
        self.cache = {}
        st = B.states.str.split("|")
        self.members = {}
        for pos, lst in zip(range(len(B)), st):
            for s in lst:
                if s:
                    self.members.setdefault(s, []).append(pos)
        self.unknown = np.flatnonzero(B.states.to_numpy() == "")

    def _build(self, key, pos):
        if key in self.cache:
            return self.cache[key]
        pos = np.asarray(pos, dtype=np.int64)
        entry = {"pos": pos}
        for col, mk in (("name_skc", _vec_name), ("addr_sk", _vec_addr)):
            try:
                v = mk(); M = v.fit_transform(self.B[col].to_numpy()[pos])
                entry[col] = (v, M.T.tocsr())
            except ValueError:            # empty vocabulary, e.g. every address blank
                entry[col] = None
        self.cache[key] = entry
        return entry

    def query(self, A, thr=0.1, chunk=40_000):
        """Returns DataFrame(a, b, sim_name, sim_addr); a = row in A, b = row in self.B."""
        out = []
        ast = A.states.str.split("|").tolist()
        groups = {}
        for i, lst in enumerate(ast):
            for s in lst:
                groups.setdefault(s, []).append(i)
        jobs = []
        for s, rows in groups.items():
            if s == "":
                jobs.append(("__all__", np.arange(len(self.B)), rows, self.k_state))
            elif s in self.members:
                jobs.append((s, self.members[s], rows, self.k_state))
        known = [i for i, lst in enumerate(ast) if lst != [""]]
        if len(self.unknown) and known:
            jobs.append(("__unknown__", self.unknown, known, self.k_unk))
        for key, pos, rows, k in jobs:
            e = self._build(key, pos)
            rows = np.asarray(rows)
            for col, tag in (("name_skc", 0), ("addr_sk", 1)):
                if e[col] is None:
                    continue
                v, BT = e[col]
                texts = A[col].to_numpy()[rows]
                for i in range(0, len(rows), chunk):
                    C = sp_matmul_topn(v.transform(texts[i:i + chunk]), BT, top_n=k, threshold=thr,
                                       n_threads=N_CPU).tocoo()
                    out.append((rows[i + C.row], e["pos"][C.col], C.data, tag))
        if not out:
            return pd.DataFrame({"a": [], "b": [], "sim_name": [], "sim_addr": []})
        a = np.concatenate([o[0] for o in out]).astype(np.int64)
        b = np.concatenate([o[1] for o in out]).astype(np.int64)
        sim = np.concatenate([o[2] for o in out]).astype(np.float32)
        tag = np.concatenate([np.full(len(o[0]), o[3], np.int8) for o in out])
        df = pd.DataFrame({"key": a * len(self.B) + b,
                           "sim_name": np.where(tag == 0, sim, np.nan),
                           "sim_addr": np.where(tag == 1, sim, np.nan)})
        df = df.groupby("key", sort=False).max().reset_index()
        df["a"] = (df.key // len(self.B)).astype(np.int32)
        df["b"] = (df.key % len(self.B)).astype(np.int32)
        return df.drop(columns="key")

# ---------------------------------------------------------------------
# 4. Features: how similar is each (Source 1, candidate) pair?
# ---------------------------------------------------------------------
FULL = [("tset", fuzz.token_set_ratio), ("tsort", fuzz.token_sort_ratio), ("ratio", fuzz.ratio),
        ("partial", fuzz.partial_ratio), ("jw", JaroWinkler.normalized_similarity)]

def _nums(s):
    return set(re.findall(r"\d+", s))

def features(P, A, B, is_s3):
    a, b = P.a.to_numpy(), P.b.to_numpy()
    g = lambda D, col, idx: D[col].to_numpy()[idx]
    f = pd.DataFrame({"sim_name": P.sim_name.to_numpy(), "sim_addr": P.sim_addr.to_numpy()})
    for col, pre, scorers in (("name_b", "nb", FULL), ("name_sk", "nsk", FULL),
                              ("name_skc", "nskc", [("ratio", fuzz.ratio), ("partial", fuzz.partial_ratio)]),
                              ("addr_sk", "ask", [("tset", fuzz.token_set_ratio), ("ratio", fuzz.ratio),
                                                  ("partial", fuzz.partial_ratio)])):
        x, y = g(A, col, a).tolist(), g(B, col, b).tolist()
        for nm, sc in scorers:
            f[f"{pre}_{nm}"] = cpdist(x, y, scorer=sc, workers=N_CPU).astype(np.float32)
    xa, ya = g(A, "addr_n", a), g(B, "addr_n", b)
    nl = [_nums(s) for s in xa]; nr = [_nums(s) for s in ya]
    f["num_jacc"] = np.array([len(p & q) / len(p | q) if (p | q) else -1 for p, q in zip(nl, nr)], np.float32)
    f["num_contain"] = np.array([len(p & q) / len(q) if q else -1 for p, q in zip(nl, nr)], np.float32)
    sa, sb = g(A, "states", a), g(B, "states", b)
    f["state_rel"] = np.array([-1 if (not p or not q) else (1 if set(p.split("|")) & set(q.split("|")) else 0)
                               for p, q in zip(sa, sb)], np.int8)
    f["name_exact"] = (g(A, "name_skc", a) == g(B, "name_skc", b)).astype(np.int8)
    f["cand_addr_len"] = np.fromiter((len(s) for s in ya), np.int32, len(ya))
    f["s1_addr_len"] = np.fromiter((len(s) for s in xa), np.int32, len(xa))
    f["cand_name_len"] = np.fromiter((len(s) for s in g(B, "name_b", b)), np.int32, len(b))
    f["s1_name_len"] = np.fromiter((len(s) for s in g(A, "name_b", a)), np.int32, len(a))
    f["cand_nonascii"] = g(B, "nonascii", b)
    f["is_s3"] = np.int8(is_s3)
    # context: how this candidate compares with the other candidates of the same Source 1 record
    v = f.nsk_tset.to_numpy() + f.ask_tset.to_numpy()
    grp = pd.DataFrame({"a": a, "v": v}).groupby("a")["v"]
    f["rank_s1"] = grp.rank(ascending=False, method="min").to_numpy(np.float32)
    f["gap_s1"] = (grp.transform("max").to_numpy() - v).astype(np.float32)
    f["n_cand_s1"] = grp.transform("size").to_numpy(np.int32)
    return f

# ---------------------------------------------------------------------
# 5. Scoring (same formula as the leaderboard) and the final decision rule
# ---------------------------------------------------------------------
def macro_f05(pred, truth, ids):
    sc = []
    for s in ids:
        p, t = pred.get(s, set()), truth.get(s, set())
        if not t:
            sc.append(1.0 if not p else 0.0); continue
        tp = len(p & t)
        if tp == 0:
            sc.append(0.0); continue
        P, R = tp / len(p), tp / len(t)
        sc.append(1.25 * P * R / (0.25 * P + R))
    return float(np.mean(sc))

def decide(pairs, thr):
    """pairs: DataFrame(s1_id, cand_id, p). Keep p >= thr; each S2/S3 record goes to at most one S1 (its best)."""
    d = pairs[pairs.p >= thr].sort_values("p", ascending=False).drop_duplicates("cand_id")
    return d.groupby("s1_id")["cand_id"].apply(set).to_dict()

def write_tsv(path, s1_ids, mapping, col):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            fh.write(s + "\t" + ",".join(sorted(mapping.get(s, ()))) + "\n")


# ======================================================================
# 2. v8 additions: rarity + phonetic + rank features, models
# ======================================================================
# =====================================================================
# v8 additions: richer pair features + an ensemble of gradient boosting models.
# Everything here works on top of the v1 pipeline (erpipe) and uses no external data.
# Libraries: jellyfish (MIT), xgboost (Apache-2.0), catboost (Apache-2.0), lightgbm (MIT).
# =====================================================================
import jellyfish
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from catboost import CatBoostClassifier

# ---------------------------------------------------------------------
# 1. Extra cleaned columns per record
#    name_w  : the name's words without legal words (Pvt, Ltd, LLC, SARL ...)
#    name_ph : the same words as sound codes (metaphone), so "Sharma" and "Sharmaa" give the same code
# ---------------------------------------------------------------------
def _extra_rows(names):
    out = []
    for nb in names:
        toks = nb.split()
        core = [t for t in toks if skel_tok(t) not in LEGAL_SK] or toks
        ph = [jellyfish.metaphone(t) if not t.isdigit() else t for t in core]
        out.append((" ".join(core), " ".join(p for p in ph if p)))
    return out

def prepare_extra(df):
    names = df.name_b.tolist(); step = 100_000
    with Pool(N_CPU) as pool:
        res = pool.map(_extra_rows, [names[i:i + step] for i in range(0, len(names), step)])
    res = [r for chunk in res for r in chunk]
    df["name_w"] = [r[0] for r in res]
    df["name_ph"] = [r[1] for r in res]
    return df

# ---------------------------------------------------------------------
# 2. Word-rarity ("IDF") weights, learned from the candidate pool itself.
#    A word shared by 50,000 businesses ("store", "road") says little; a rare word ("zanskar") says a lot.
# ---------------------------------------------------------------------
class Rarity:
    """Word vocabularies + rarity weights for one candidate pool (one source, one country)."""
    def __init__(self, B):
        self.parts = {}
        for col in ("name_w", "addr_n", "name_ph"):
            v = CountVectorizer(token_pattern=r"\S+", binary=True, dtype=np.float32)
            try:
                X = v.fit_transform(B[col].to_numpy())
            except ValueError:            # empty vocabulary
                self.parts[col] = None; continue
            df = np.asarray(X.sum(axis=0)).ravel()
            idf = (np.log((len(B) + 1) / (df + 1)) + 1).astype(np.float32)
            self.parts[col] = (v, idf, X.tocsr(), float(np.log(len(B) + 1) + 1))
        # full char 3-gram similarity of the name skeleton for EVERY pair (the shortlist only kept top-15)
        self.cv = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), dtype=np.float32, sublinear_tf=True)
        try:
            self.CB = self.cv.fit_transform(B.name_skc.to_numpy()).tocsr()
        except ValueError:
            self.cv = None

def _rowdot(XA, XB):
    return np.asarray(XA.multiply(XB).sum(axis=1)).ravel().astype(np.float32)

def _rowmax(X):
    if X.nnz == 0:
        return np.zeros(X.shape[0], np.float32)
    return np.asarray(X.max(axis=1).todense()).ravel().astype(np.float32)

def rarity_features(R, a, b, A, f, chunk=3_000_000):
    """Adds rarity-weighted similarities for (A row a, pool row b) pairs into DataFrame f."""
    for col, pre in (("name_w", "rn"), ("addr_n", "ra"), ("name_ph", "rp")):
        names = [f"{pre}_{x}" for x in ("wjacc", "cont_s1", "cont_c", "cos", "miss_s1", "miss_c", "share_max")]
        if R.parts[col] is None:
            for n in names: f[n] = np.float32(np.nan)
            continue
        v, idf, XB_all, max_idf = R.parts[col]
        texts = A[col].to_numpy()
        XA_all = v.transform(texts).tocsr()
        # words of the S1 record that never occur in the pool: maximally rare, and surely not shared
        n_unseen = np.array([len(set(t.split())) for t in texts], np.float32) - np.asarray(XA_all.sum(axis=1)).ravel()
        W = sp.diags(idf); W2 = sp.diags(idf * idf)
        XAw, XBw = (XA_all @ W).tocsr(), (XB_all @ W).tocsr()
        sA = np.asarray(XAw.sum(axis=1)).ravel() + n_unseen * max_idf
        sB = np.asarray(XBw.sum(axis=1)).ravel()
        nA = np.sqrt(np.asarray((XA_all @ W2).sum(axis=1)).ravel() + n_unseen * max_idf ** 2)
        nB = np.sqrt(np.asarray((XB_all @ W2).sum(axis=1)).ravel())
        res = {n: np.empty(len(a), np.float32) for n in names}
        for i in range(0, len(a), chunk):
            ai, bi = a[i:i + chunk], b[i:i + chunk]
            PA, PB = XAw[ai], XB_all[bi]
            S = PA.multiply(PB).tocsr()                               # shared words, with their weights
            inter = np.asarray(S.sum(axis=1)).ravel()
            inter2 = np.asarray(S.multiply(S).sum(axis=1)).ravel()
            sa, sb = sA[ai], sB[bi]
            with np.errstate(divide="ignore", invalid="ignore"):
                res[names[0]][i:i + chunk] = np.where(sa + sb - inter > 0, inter / (sa + sb - inter), np.nan)
                res[names[1]][i:i + chunk] = np.where(sa > 0, inter / sa, np.nan)
                res[names[2]][i:i + chunk] = np.where(sb > 0, inter / sb, np.nan)
                res[names[3]][i:i + chunk] = np.where(nA[ai] * nB[bi] > 0, inter2 / (nA[ai] * nB[bi]), np.nan)
            missA = _rowmax((PA - PA.multiply(PB > 0)).tocsr())        # rarest S1 word the candidate lacks
            res[names[4]][i:i + chunk] = np.maximum(missA, np.where(n_unseen[ai] > 0, max_idf, 0))
            PBw = XBw[bi]
            res[names[5]][i:i + chunk] = _rowmax((PBw - PBw.multiply(PA > 0)).tocsr())   # rarest candidate word S1 lacks
            res[names[6]][i:i + chunk] = _rowmax(S)                   # rarest word they share
        for n in names: f[n] = res[n]
    if R.cv is not None:
        CA = R.cv.transform(A.name_skc.to_numpy()).tocsr()
        out = np.empty(len(a), np.float32)
        for i in range(0, len(a), chunk):
            out[i:i + chunk] = _rowdot(CA[a[i:i + chunk]], R.CB[b[i:i + chunk]])
        f["nch_cos"] = out
    else:
        f["nch_cos"] = np.float32(np.nan)

def phonetic_features(a, b, A, B, f):
    x, y = A.name_ph.to_numpy()[a].tolist(), B.name_ph.to_numpy()[b].tolist()
    f["ph_tset"] = cpdist(x, y, scorer=fuzz.token_set_ratio, workers=N_CPU).astype(np.float32)
    f["ph_ratio"] = cpdist(x, y, scorer=fuzz.ratio, workers=N_CPU).astype(np.float32)
    f["ph_first_eq"] = np.array([(p.split(" ", 1)[0] == q.split(" ", 1)[0]) if p and q else -1
                                 for p, q in zip(x, y)], np.int8)
    xw, yw = A.name_w.to_numpy()[a].tolist(), B.name_w.to_numpy()[b].tolist()
    f["nw_tset"] = cpdist(xw, yw, scorer=fuzz.token_set_ratio, workers=N_CPU).astype(np.float32)

# ---------------------------------------------------------------------
# 3. Rank features: how does this candidate compare with the business's other candidates (same source)?
# ---------------------------------------------------------------------
RANK_ON = ["rn_cos", "ra_cos", "nch_cos", "ph_tset", "combo"]

def rank_features(a, f):
    f["combo"] = (f.rn_cos.fillna(0) + f.ra_cos.fillna(0) + f.nch_cos.fillna(0) + f.ph_tset / 100).astype(np.float32)
    g = pd.DataFrame({"a": a})
    for k in RANK_ON:
        g["v"] = f[k].fillna(-1).to_numpy()
        grp = g.groupby("a")["v"]
        mx = grp.transform("max").to_numpy()
        f[f"rk_{k}"] = grp.rank(ascending=False, method="min").to_numpy(np.float32)
        f[f"gap_{k}"] = (mx - g.v.to_numpy()).astype(np.float32)
    v = g.v.to_numpy()                      # g.v is "combo" now (last in RANK_ON)
    # runner-up value per business, found by sorting once
    order = np.lexsort((-v, a)); a_s, v_s = a[order], v[order]
    first = np.r_[True, a_s[1:] != a_s[:-1]]
    pos = np.arange(len(a_s)); start = np.maximum.accumulate(np.where(first, pos, 0))
    has2 = (start + 1 < len(a_s)) & (a_s[np.minimum(start + 1, len(a_s) - 1)] == a_s)
    sec_s = np.where(has2, v_s[np.minimum(start + 1, len(a_s) - 1)], -1)
    second = np.empty_like(v); second[order] = sec_s
    # top candidate: lead over the runner-up; others: distance behind the leader (negative)
    f["combo_lead"] = np.where(v >= mx, v - second, v - mx).astype(np.float32)
    f["combo_z"] = ((g.v - grp.transform("mean")) / (grp.transform("std").fillna(0) + 1e-3)).to_numpy(np.float32)
    f["n_close"] = g.assign(c=(mx - v) < 0.15).groupby("a")["c"].transform("sum").to_numpy(np.int16)

def features_v8(P, A, B, is_s3, R):
    """v1 features + v8 features for the shortlist pairs P of A (S1 batch) against pool B."""
    f = features(P, A, B, is_s3)
    a, b = P.a.to_numpy(), P.b.to_numpy()
    rarity_features(R, a, b, A, f)
    phonetic_features(a, b, A, B, f)
    rank_features(a, f)
    return f

# ---------------------------------------------------------------------
# 4. The ensemble: several boosting models, each gives a probability; the blend is their average.
# ---------------------------------------------------------------------
MODEL_SPECS = [
    ("lgb_a", "lgb", dict(learning_rate=0.05, num_leaves=255, min_child_samples=100, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=0)),
    ("lgb_b", "lgb", dict(learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.7, subsample_freq=1,
                          colsample_bytree=0.5, reg_lambda=5.0, random_state=1)),
    ("xgb", "xgb", dict(eta=0.08, max_depth=9, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                        tree_method="hist", max_bin=256, eval_metric="logloss", objective="binary:logistic", seed=2)),
    ("cat", "cat", dict(learning_rate=0.12, depth=8, l2_leaf_reg=3, random_seed=3, border_count=128)),
]

def fit_model(kind, params, Xtr, ytr, Xva, yva, max_rounds, n_rounds=None):
    """Trains one model. With n_rounds=None it stops when the held-out loss stops improving."""
    es = n_rounds is None
    rounds = max_rounds if es else n_rounds
    if kind == "lgb":
        m = lgb.LGBMClassifier(n_estimators=rounds, n_jobs=N_CPU, verbose=-1, **params)
        m.fit(Xtr, ytr, **({"eval_set": [(Xva, yva)], "callbacks": [lgb.early_stopping(100, verbose=False)]} if es else {}))
        return m, (m.best_iteration_ if es else rounds)
    if kind == "xgb":
        dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=params.get("max_bin", 256), nthread=N_CPU)
        ev = [(xgb.DMatrix(Xva, yva, nthread=N_CPU), "val")] if es else []
        m = xgb.train({**params, "nthread": N_CPU}, dtr, rounds, evals=ev,
                      early_stopping_rounds=100 if es else None, verbose_eval=False)
        del dtr; gc.collect()
        return m, (m.best_iteration + 1 if es else rounds)
    if kind == "cat":
        m = CatBoostClassifier(iterations=rounds, thread_count=N_CPU, verbose=False, allow_writing_files=False,
                               **({"od_type": "Iter", "od_wait": 100, "use_best_model": True} if es else {}), **params)
        m.fit(Xtr, ytr, eval_set=(Xva, yva) if es else None)
        return m, (m.get_best_iteration() + 1 if es else rounds)

def predict_model(kind, m, X):
    if kind == "lgb":
        return m.predict(X, num_threads=N_CPU).astype(np.float32) if isinstance(m, lgb.Booster) \
            else m.predict_proba(X)[:, 1].astype(np.float32)
    if kind == "xgb":
        return m.predict(xgb.DMatrix(X, nthread=N_CPU)).astype(np.float32)
    return m.predict_proba(X)[:, 1].astype(np.float32)

def save_model(kind, m, path):
    if kind == "lgb": m.booster_.save_model(path)
    elif kind == "xgb": m.save_model(path)
    else: m.save_model(path)

def load_model(kind, path):
    if kind == "lgb": return lgb.Booster(model_file=path)
    if kind == "xgb":
        m = xgb.Booster(); m.load_model(path); m.set_param({"nthread": N_CPU}); return m
    m = CatBoostClassifier(); m.load_model(path); return m

MODEL_EXT = {"lgb": "txt", "xgb": "json", "cat": "cbm"}


# ======================================================================
# 3. v7 exact checks (house number, legal form)
# ======================================================================
import ctypes
def free_mem():
    """Free memory and hand it back to the system (so helper processes don't copy it)."""
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

# ---------------------------------------------------------------------
# v7 exactness checks. The data contains planted look-alikes: same name and street, but a different
# house number (712 vs 719) or a different legal form (LLP vs Private Limited).
# ---------------------------------------------------------------------
LEGAL_MAP = {"prwt": "pvt", "pwt": "pvt", "prvt": "pvt", "lmtd": "ltd", "ltd": "ltd", "lmt": "ltd", "llp": "llp",
             "llc": "llc", "inc": "inc", "ncrprtd": "inc", "crp": "corp", "crprtn": "corp", "cmpny": "co", "c": "co",
             "plc": "plc", "pc": "pc", "pllc": "pllc", "lp": "lp", "srl": "sarl", "eurl": "eurl", "ss": "sas",
             "ssu": "sas", "sa": "sa", "gmbh": "gmbh", "snc": "snc", "sc": "sci", "sk": "sci", "dba": "dba"}

def _legal_rows(rows):
    out = []
    for nb, an in rows:
        lg = set(LEGAL_MAP.get(skel_tok(t)) for t in nb.split()) - {None}
        if "pvt" in lg: lg.discard("ltd")                  # "Private Limited" is one legal form
        nums = [x.lstrip("0") or "0" for x in re.findall(r"\d+", an)]
        out.append(("|".join(sorted(lg)), " ".join(nums)))
    return out

def add_legal(D):
    """Adds per-record columns: legal (legal form, e.g. 'pvt', 'llp|co') and nums (numbers in the address)."""
    rows = list(zip(D.name_b.tolist(), D.addr_n.tolist())); step = 100_000
    with Pool(N_CPU) as pool:
        res = [r for ch in pool.map(_legal_rows, [rows[i:i + step] for i in range(0, len(rows), step)]) for r in ch]
    D["legal"] = [r[0] for r in res]; D["nums"] = [r[1] for r in res]
    return D

def _near(x, y):
    """Two different numbers that look like a planted change: close in value, or same length and one digit apart."""
    if x == y: return False
    if abs(int(x[:9]) - int(y[:9])) <= 20: return True
    return len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1

EXACT_FEATS = ["legal_rel", "legal_s1", "legal_cand", "num_eq", "num_cand_extra", "num_s1_extra", "num_first_eq",
               "num_near", "x_noextra", "x_legal_same"]

def _exact_rows(args):
    out = np.zeros((len(args[0]), 8), np.int16)
    for i, (x, y, p, q) in enumerate(zip(*args)):
        lr = -1 if (not x or not y) else int(x == y)
        pl_, ql = p.split(), q.split(); ps, qs = set(pl_), set(ql)
        ea, eb = ps - qs, qs - ps
        first = -1 if (not pl_ or not ql) else int(pl_[0] == ql[0])
        near = int(any(_near(u, v) for u in ea for v in eb)) if ea and eb else 0
        out[i] = (lr, int(bool(x)), int(bool(y)), int(ps == qs), len(eb), len(ea), first, near)
    return out

def exact_features(P, A, B):
    a, b = P.a.to_numpy(), P.b.to_numpy()
    cols = [A.legal.to_numpy()[a], B.legal.to_numpy()[b], A.nums.to_numpy()[a], B.nums.to_numpy()[b]]
    step = 200_000
    with Pool(N_CPU) as pool:
        res = pool.map(_exact_rows, [[c[i:i + step].tolist() for c in cols] for i in range(0, len(P), step)])
    F = pd.DataFrame(np.concatenate(res) if res else np.zeros((0, 8), np.int16), columns=EXACT_FEATS[:8])
    F["x_noextra"] = ((F.num_cand_extra.to_numpy() == 0) & (pd.Series(cols[3]).str.len().to_numpy() > 0)).astype(np.int8)
    F["x_legal_same"] = (F.legal_rel.to_numpy() == 1).astype(np.int8)
    return F


# ======================================================================
# 4. v10 settings
# ======================================================================
import shutil
from sklearn.model_selection import GroupKFold
ROOT = os.environ.get("ROOT", "/kaggle/input")      # the notebook searches here for the .tsv files
WORK = os.environ.get("WORK", "/kaggle/working")    # models + outputs are saved here
TMP = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else WORK     # big temporary files
SMOKE = int(os.environ.get("SMOKE", 0))             # 1 = tiny quick run to check the code (20-40 min). 0 = the real run.
N_BASE = int(os.environ.get("N_BASE", 150_000))     # training businesses (v1's sample; 20% of it = the same check businesses as v1)
KEEP_P = 0.01        # stage-1 filter: candidates below this chance are dropped before stage 2
K_CTX = 8            # stage 2 compares each candidate with this many of the business's best candidates
BATCH = int(os.environ.get("BATCH", 100_000))       # test businesses processed at once (lower it if memory runs out)
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 3000))   # upper limit of trees per stage-2 model (early stopping picks the number)
# France (and any unseen country) has no answers to tune on. Its cut-off = the average of the India/US cut-offs + this margin.
FRANCE_MARGIN = float(os.environ.get("FRANCE_MARGIN", 0.0))
# Split the prediction over two sessions if needed, e.g. ONLY_COUNTRIES = "France,US" in one and "India" in the other.
ONLY_COUNTRIES = [c for c in os.environ.get("ONLY_COUNTRIES", "").split(",") if c]
os.makedirs(WORK, exist_ok=True); os.makedirs(TMP, exist_ok=True)
log("CPU cores:", N_CPU, "| training businesses:", N_BASE, "| smoke test:", bool(SMOKE), "| only countries:", ONLY_COUNTRIES or "all")


# ======================================================================
# 5. v10 helpers (stage 2, ids, scoring)
# ======================================================================
# =====================================================================
# v10 helpers.
# Stage 1 scores each (business, candidate) pair alone with the strong v8 features (+ exact house-number / legal-form checks).
# Stage 2 looks at ALL candidates of one business together (Source 2 and Source 3): one real business shows up about
# 3 times and those copies look alike, so a candidate that resembles the other strong candidates gets a boost.
# Idea and stage-2 features from v2 (which gained +0.004 over v1), here on top of a much stronger stage 1.
# =====================================================================
GRID = np.round(np.arange(0.30, 0.9501, 0.025), 3)     # cut-offs tried
REL_GRID = (0.0, 0.5, 0.7, 0.85)                        # "keep only candidates within this share of the business's best"

def add_name_freq(D):
    """How many records per 100k in the same country share this cleaned name (generic names are common, esp. in France)."""
    g = D.groupby(["country", "name_skc"]).name_skc.transform("size")
    D["name_freq"] = (g / D.groupby("country").name_skc.transform("size") * 1e5).astype(np.float32)
    return D

def prep_all(df):
    """All cleaned columns of a raw table: v1 cleaning, v8 sound codes, v7 legal form + numbers, name commonness."""
    return add_name_freq(add_legal(prepare_extra(prepare(df))))

def features_v10(P, A, B, is_s3, R):
    """Stage-1 features: v1 + v8 features, plus v7's exact checks (house number, legal form), plus name commonness."""
    f = features_v8(P, A, B, is_s3, R)
    a, b = P.a.to_numpy(), P.b.to_numpy()
    E = exact_features(P, A, B)
    for c in EXACT_FEATS:
        f[c] = E[c].to_numpy()
    f["s1_name_freq"] = A.name_freq.to_numpy()[a]
    f["cand_name_freq"] = B.name_freq.to_numpy()[b]
    f["n_name_close"] = pd.Series(f.nskc_ratio.to_numpy() >= 90).groupby(a).transform("sum").to_numpy()
    return f

S2_EXTRA = ["p1", "p1_rank", "p1_max", "p1_2nd", "p1_sum", "p1_rel", "n_hi", "n_kept",
            "sup_max", "sup_sum", "sup_cnt", "sup_name_max", "sup_addr_max", "sup_other_src"]

def stage2_features(key, p1, cname, caddr, is_s3, K=K_CTX):
    """Context features. key: business of each row; p1: stage-1 chance; cname/caddr: candidate's cleaned text."""
    d = pd.DataFrame({"key": key, "p1": p1.astype(np.float32), "row": np.arange(len(key)), "s3": is_s3})
    d = d.sort_values(["key", "p1"], ascending=[True, False], kind="stable")
    g = d.groupby("key", sort=False)
    d["p1_rank"] = (g.cumcount() + 1).astype(np.float32)
    d["p1_max"] = g.p1.transform("max")
    d["p1_2nd"] = d.key.map(d[d.p1_rank == 2].set_index("key").p1).fillna(0).to_numpy(np.float32)
    d["p1_sum"] = g.p1.transform("sum")
    d["p1_rel"] = d.p1 / d.p1_max.clip(lower=1e-6)
    d["n_hi"] = (d.p1 > 0.5).groupby(d.key).transform("sum").astype(np.float32)
    d["n_kept"] = g.p1.transform("size").astype(np.float32)
    top = d[d.p1_rank <= K][["key", "row", "p1", "s3"]]
    j = top.merge(top, on="key", suffixes=("", "_o"))
    j = j[j.row != j.row_o]
    r, ro = j.row.to_numpy(), j.row_o.to_numpy()
    sn = cpdist(cname[r].tolist(), cname[ro].tolist(), scorer=fuzz.token_set_ratio, workers=N_CPU).astype(np.float32) / 100
    sa = cpdist(caddr[r].tolist(), caddr[ro].tolist(), scorer=fuzz.token_set_ratio, workers=N_CPU).astype(np.float32) / 100
    po = j.p1_o.to_numpy()
    s = (sn + sa) / 2
    j = pd.DataFrame({"row": r, "w": po * s, "strong": ((s > 0.85) & (po > 0.5)).astype(np.int8),
                      "wn": po * sn, "wa": po * sa, "wx": np.where(j.s3.to_numpy() != j.s3_o.to_numpy(), po * s, 0)})
    agg = j.groupby("row").agg(sup_max=("w", "max"), sup_sum=("w", "sum"), sup_cnt=("strong", "sum"),
                               sup_name_max=("wn", "max"), sup_addr_max=("wa", "max"), sup_other_src=("wx", "max"))
    d = d.set_index("row").sort_index().join(agg)
    return d[S2_EXTRA].fillna(0).astype(np.float32).reset_index(drop=True)

def stage2_in_chunks(key, *cols, chunk=200_000):
    """stage2_features for many businesses, a block of businesses at a time to save memory."""
    order = np.argsort(key, kind="stable"); ks = key[order]
    uk = np.unique(ks); out = []
    for i in range(0, len(uk), chunk):
        lo, hi = np.searchsorted(ks, uk[i]), np.searchsorted(ks, uk[min(i + chunk, len(uk)) - 1], side="right")
        sel = order[lo:hi]
        f = stage2_features(key[sel], *[c[sel] for c in cols]); f.index = sel; out.append(f)
    return pd.concat(out).sort_index()

def f05_rows(npred, tp, ntrue):
    """F0.5 of every Source 1 record from counts (same formula as the leaderboard; empty-and-right = 1)."""
    P = tp / np.maximum(npred, 1); R = tp / np.maximum(ntrue, 1)
    f = np.where(tp > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0.0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f)

def id_codes(ids):
    """'S2-123' -> 246, 'S3-123' -> 247 (all ids are plain numbers, checked on the real files)."""
    return np.array([int(x[3:]) * 2 + (x[1] == "3") for x in ids], np.int64)

def decode_expr(col="cand"):
    """polars expression that turns the code back into 'S2-123' / 'S3-123'."""
    return (pl.lit("S") + (2 + pl.col(col) % 2).cast(pl.Utf8) + pl.lit("-") + (pl.col(col) // 2).cast(pl.Utf8))

def restore_models():
    """Two-session runs: if this session has no trained models, copy them from an earlier notebook version
    that is attached as an input (its Output files show up under /kaggle/input/...)."""
    if os.path.exists(f"{WORK}/v10_meta.json"):
        return
    hits = glob.glob(f"{ROOT}/**/v10_meta.json", recursive=True)
    if not hits:
        raise FileNotFoundError("v10_meta.json not found: run the train stage first, or add the earlier notebook version "
                                "(the one that ran RUN = 'train') as an input via Add Input -> Notebook Output")
    for f in glob.glob(os.path.dirname(hits[0]) + "/v10_*"):
        shutil.copy(f, WORK)
    log("copied trained models from", os.path.dirname(hits[0]))

def part_dirs():
    """Folders holding finished prediction batches: this session's, and those of earlier sessions attached as inputs."""
    return [f"{WORK}/parts_v10"] + sorted(set(os.path.dirname(p) for p in glob.glob(f"{ROOT}/**/parts_v10/*.parquet", recursive=True)))

def part_files():
    """All finished batch files (one per name; this session's copy wins)."""
    seen = {}
    for d in part_dirs():
        for f in sorted(glob.glob(f"{d}/*.parquet")):
            seen.setdefault(os.path.basename(f), f)
    return sorted(seen.values())


# ======================================================================
# 6. TRAIN
# ======================================================================
if 'train' in STAGES:
    # ---------------------------------------------------------------------
    # TRAIN v10. Same held-out check businesses as v1/v6/v7/v8/v9 (20% of v1's 150k sample), so scores compare directly.
    #  1. shortlist + stage-1 features for the training businesses (Source 2 and Source 3 together, country by country)
    #  2. stage 1: out-of-fold chances (each business is scored by a model that never saw it), then one full stage-1 model
    #  3. stage-2 features from those chances, then a team of 3 stage-2 models, checked on the held-out businesses
    #  4. cut-off + "close to the best" share per country
    # ---------------------------------------------------------------------
    T0 = time.time()
    log("loading train Source 1 + ground truth")
    s1 = load(ROOT, "train_source1"); gt = load(ROOT, "train_ground_truth")
    truth = {r.source1_entity_id: set(x for x in r.matched_entity_ids.split(",") if x) for r in gt.itertuples()}
    del gt; gc.collect()
    s1 = s1.sample(4000 if SMOKE else min(N_BASE, len(s1)), random_state=0).reset_index(drop=True)
    A = prep_all(s1); del s1; gc.collect(); log("prepared S1 sample", A.shape)
    raw2, raw3 = load(ROOT, "train_source2"), load(ROOT, "train_source3")
    gsets = {s: truth[s] for s in A.entity_id}                      # only the sampled businesses
    del truth; gc.collect()

    Fs, Ms = [], []
    for country in A.country.unique():
        Ac = A[A.country == country].reset_index(drop=True)
        pools = []
        for raw, is_s3 in ((raw2, 0), (raw3, 1)):
            R_ = raw[raw.country == country]
            if SMOKE:   # keep the true matches of the sample + 100k random records, so the shortlist still has look-alikes
                want = set().union(*(gsets[s] for s in Ac.entity_id))
                R_ = pd.concat([R_[R_.entity_id.isin(want)], R_.sample(min(100_000, len(R_)), random_state=0)]).drop_duplicates("entity_id")
            pools.append((prep_all(R_.reset_index(drop=True)), is_s3)); del R_; free_mem()
        for Bc, is_s3 in pools:
            P = Index(Bc).query(Ac)
            if len(P) == 0: continue
            R = Rarity(Bc); F = features_v10(P, Ac, Bc, is_s3, R); del R
            m = pd.DataFrame({"s1_id": Ac.entity_id.to_numpy()[P.a.to_numpy()], "cand_id": Bc.entity_id.to_numpy()[P.b.to_numpy()],
                              "cname": Bc.name_sk.to_numpy()[P.b.to_numpy()], "caddr": Bc.addr_sk.to_numpy()[P.b.to_numpy()]})
            m["s3"] = np.int8(is_s3)
            Fs.append(F.astype(np.float32)); Ms.append(m)
            log(country, "Source", 3 if is_s3 else 2, "pairs", len(m), "| minutes so far", round((time.time() - T0) / 60, 1))
            del P, F, m; free_mem()
        del pools, Ac; free_mem()
    del raw2, raw3; free_mem()
    X1 = pd.concat(Fs, ignore_index=True); M = pd.concat(Ms, ignore_index=True); del Fs, Ms; gc.collect()
    FEAT1 = list(X1.columns)
    Xn = X1.to_numpy(np.float32); del X1; gc.collect()
    y = np.fromiter((c in gsets[s] for s, c in zip(M.s1_id, M.cand_id)), bool, len(M)).astype(np.int8)
    all_true = sum(len(v) for v in gsets.values())
    log("pairs", len(M), "per S1", round(len(M) / len(A), 1), "| shortlist recall", round(y.sum() / all_true, 4),
        "| stage-1 features", len(FEAT1), "(v8 shortlist recall was about 0.9765)")

    # ---------- stage 1: out-of-fold chances ----------
    P1 = dict(n_estimators=400, learning_rate=0.08, num_leaves=255, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, n_jobs=N_CPU, verbose=-1)
    if SMOKE: P1["n_estimators"] = 60
    gcode = pd.factorize(M.s1_id)[0]
    p1 = np.zeros(len(M), np.float32)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=3).split(Xn, y, groups=gcode)):
        m1 = lgb.LGBMClassifier(**P1).fit(Xn[tr], y[tr]); p1[te] = m1.predict_proba(Xn[te])[:, 1]
        log("stage-1 fold", k + 1, "of 3 done | minutes so far", round((time.time() - T0) / 60, 1)); del m1; gc.collect()
    m1 = lgb.LGBMClassifier(**P1).fit(Xn, y); m1.booster_.save_model(f"{WORK}/v10_stage1.txt"); del m1; gc.collect()
    log("stage-1 full model saved")

    # ---------- keep only candidates with a real chance, then build the stage-2 features ----------
    keep = p1 >= KEEP_P
    log("kept after stage 1:", int(keep.sum()), "pairs,", round(keep.sum() / len(A), 1), "per S1 | true matches kept",
        round(y[keep].sum() / all_true, 4))
    S2F = stage2_in_chunks(gcode[keep], p1[keep], M.cname.to_numpy()[keep], M.caddr.to_numpy()[keep], M.s3.to_numpy()[keep])
    X2 = np.hstack([Xn[keep], S2F.to_numpy(np.float32)]); del Xn, S2F; gc.collect()
    FEAT2 = FEAT1 + S2_EXTRA
    M = M[keep][["s1_id", "cand_id"]].reset_index(drop=True); y = y[keep]; p1 = p1[keep]; gc.collect()
    log("stage-2 matrix", X2.shape)

    # ---------- held-out check businesses (same as v1) ----------
    val_ids = set(np.random.default_rng(42).choice(A.entity_id.to_numpy(), int(0.2 * len(A)), replace=False))
    ctry_of = dict(zip(A.entity_id, A.country))
    vl = list(val_ids); vcountry = np.array([ctry_of[s] for s in vl])
    nv = len(vl); vpos = {s: i for i, s in enumerate(vl)}; ntrue_v = np.array([len(gsets[s]) for s in vl])
    iv = M.s1_id.isin(val_ids).to_numpy()
    Mv = M[iv][["s1_id", "cand_id"]].reset_index(drop=True); Mv["si"] = Mv.s1_id.map(vpos).to_numpy(); yva = y[iv]
    COUNTRIES = sorted(set(vcountry))

    def prep_val(p):
        """Each Source 2/3 record goes to its most likely business only; then thresholds are just filters."""
        V = Mv[["si", "cand_id"]].assign(p=p, label=yva.astype(bool))
        V = V.sort_values("p", ascending=False).drop_duplicates("cand_id")
        V["pmax"] = V.groupby("si").p.transform("max")
        return V.reset_index(drop=True)

    def val_f(V, thr, rel):
        """F0.5 of every check business for cut-off thr and 'within rel x the business's best' rule."""
        pv, pm = V.p.to_numpy(), V.pmax.to_numpy()
        m = (pv >= thr) & (pv >= rel * pm)
        s = V.si.to_numpy()[m]; lab = V.label.to_numpy()[m]
        return f05_rows(np.bincount(s, minlength=nv), np.bincount(s[lab], minlength=nv), ntrue_v)

    def tune(V, rels=(0.0,), mask=None):
        best = (-1.0, None, None)
        for rel in rels:
            for t in GRID:
                f = val_f(V, t, rel); s = float(f.mean() if mask is None else f[mask].mean())
                if s > best[0]: best = (s, float(t), float(rel))
        return best

    log("baseline all-empty", round(float(f05_rows(np.zeros(nv), np.zeros(nv), ntrue_v).mean()), 4))
    Vp = Mv[["si"]].assign(label=yva.astype(bool)); Vp = Vp[Vp.label]
    log("ceiling (perfect model on this shortlist)",
        round(float(f05_rows(np.bincount(Vp.si, minlength=nv), np.bincount(Vp.si, minlength=nv), ntrue_v).mean()), 4), "(v8 ceiling was about 0.9926)")
    r1 = tune(prep_val(p1[iv]))
    log(f"stage 1 only (like v1, but with v8 features): BEST val F0.5 = {r1[0]:.4f} at cut-off {r1[1]}")

    # ---------- stage 2: a team of 3 models ----------
    X2tr, ytr, X2va = X2[~iv], y[~iv], X2[iv]
    S2_SPECS = [s for s in MODEL_SPECS if s[0] in ("lgb_a", "lgb_b", "xgb")]
    PV, RESULTS, ROUNDS = {}, {}, {}
    for name, kind, params in S2_SPECS:
        t1 = time.time(); log("training stage-2", name)
        m, n = fit_model(kind, params, X2tr, ytr, X2va, yva, MAX_ROUNDS)
        PV[name] = predict_model(kind, m, X2va); ROUNDS[name] = n
        save_model(kind, m, f"{WORK}/v10_s2_{name}.{MODEL_EXT[kind]}")
        log(name, "trees", n, "| minutes", round((time.time() - t1) / 60, 1))
        RESULTS[name] = tune(prep_val(PV[name])); log(f"BEST val F0.5 [{name}] = {RESULTS[name][0]:.4f} at cut-off {RESULTS[name][1]}")
        del m; gc.collect()
    order = sorted(PV, key=lambda k: -RESULTS[k][0])
    cands = {"blend_all": list(PV), "best_single": order[:1]}
    for lab, members in cands.items():
        RESULTS[lab] = tune(prep_val(np.mean([PV[k] for k in members], axis=0)))
        log(f"BEST val F0.5 [{lab}: {'+'.join(members)}] = {RESULTS[lab][0]:.4f} at cut-off {RESULTS[lab][1]}")
    choice = max(cands, key=lambda k: RESULTS[k][0]); USE = cands[choice]
    V = prep_val(np.mean([PV[k] for k in USE], axis=0)); del PV; gc.collect()

    # ---------- decision rule per country ----------
    CUTS, rowF = {}, np.zeros(nv)
    for c in COUNTRIES:
        mask = vcountry == c
        s, t, r = tune(V, REL_GRID, mask); CUTS[c] = {"thr": t, "rel": r, "val": s}
        rowF[mask] = val_f(V, t, r)[mask]
        log(f"  {c}: {int(mask.sum())} check businesses -> cut-off {t}, close-to-best share {r}, val F0.5 {s:.4f}")
    overall = float(rowF.mean())
    s_one, t_one, r_one = tune(V, REL_GRID)
    GENERAL = {"thr": float(np.mean([v["thr"] for v in CUTS.values()])), "rel": float(np.mean([v["rel"] for v in CUTS.values()]))}
    log(f"FINAL choice: {choice} = {'+'.join(USE)} | BEST val F0.5 overall = {overall:.4f}  "
        f"(one shared cut-off would give {s_one:.4f}; stage 1 alone {r1[0]:.4f}) | v1 0.9529, v8 0.9673")
    json.dump({"features1": FEAT1, "features2": FEAT2, "models": [(n, k) for n, k, _ in S2_SPECS if n in USE],
               "cuts": CUTS, "general": GENERAL, "choice": choice, "results": RESULTS, "rounds": ROUNDS,
               "val_f05": overall, "stage1_only": r1[0], "keep_p": KEEP_P},
              open(f"{WORK}/v10_meta.json", "w"), indent=1)
    del X2, X2tr, X2va, V, M; gc.collect()
    log("training finished | total minutes", round((time.time() - T0) / 60, 1))


# ======================================================================
# 7. PREDICT test
# ======================================================================
if 'predict' in STAGES:
    # ---------------------------------------------------------------------
    # PREDICT: every test business, in batches saved to disk (safe to resume, also across sessions).
    # For each batch of businesses, Source 2 and Source 3 candidates are scored together, so stage 2 sees all of them.
    # Only probabilities are stored; the cut-offs are applied in the next step, so they can be changed without predicting again.
    # ---------------------------------------------------------------------
    T0 = time.time()
    restore_models()
    meta = json.load(open(f"{WORK}/v10_meta.json"))
    KEEP = meta["keep_p"]
    b1 = lgb.Booster(model_file=f"{WORK}/v10_stage1.txt")
    MODELS = [(n, k, load_model(k, f"{WORK}/v10_s2_{n}.{MODEL_EXT[k]}")) for n, k in meta["models"]]
    log("stage-2 models:", [n for n, _, _ in MODELS], "| val F0.5 (US+India check businesses):", round(meta["val_f05"], 4),
        "| stage 1 alone:", round(meta["stage1_only"], 4))
    PARTS = f"{WORK}/parts_v10"; os.makedirs(PARTS, exist_ok=True)
    DONE_DIRS = part_dirs()
    def is_done(name):
        return any(os.path.exists(f"{d}/{name}") for d in DONE_DIRS)

    log("loading test Source 1"); A_full = load(ROOT, "test_source1")
    s1_ids = A_full.entity_id.to_numpy()
    A = A_full.groupby("country").head(1500).reset_index(drop=True) if SMOKE else A_full
    del A_full; gc.collect()
    row_of = {s: i for i, s in enumerate(s1_ids)}
    A = prep_all(A); log("prepared", A.shape)
    print(A.country.value_counts().to_string())
    A_rowid = np.array([row_of[s] for s in A.entity_id], np.int32)
    raw2, raw3 = load(ROOT, "test_source2"), load(ROOT, "test_source3")
    for country in A.country.unique():                         # any label works, France included
        if ONLY_COUNTRIES and country not in ONLY_COUNTRIES:
            continue
        a_rows = np.flatnonzero(A.country.to_numpy() == country)
        starts = [i for i in range(0, len(a_rows), BATCH) if not is_done(f"{country}_{i}.parquet")]
        if not starts:
            log(country, "already done"); continue
        log(country, "-> cleaning the Source 2 and Source 3 records of this country")
        pools = []
        for raw, is_s3 in ((raw2, 0), (raw3, 1)):
            R_ = raw[raw.country == country].reset_index(drop=True)
            if SMOKE: R_ = R_.sample(min(150_000, len(R_)), random_state=0).reset_index(drop=True)
            Bc = prep_all(R_); del R_; free_mem()
            pools.append((Bc, is_s3, Index(Bc), Rarity(Bc), id_codes(Bc.entity_id.to_numpy())))
        log(country, "ready:", [len(p[0]) for p in pools], "records | batches to do:", len(starts))
        for i in starts:
            out = f"{PARTS}/{country}_{i}.parquet"
            rows = a_rows[i:i + BATCH]; Ac = A.iloc[rows].reset_index(drop=True)
            qa, cc, cn, ca, s3, Xs = [], [], [], [], [], []
            for Bc, is_s3, idx, R, code in pools:
                P = idx.query(Ac)
                if len(P) == 0: continue
                b = P.b.to_numpy()
                Xs.append(features_v10(P, Ac, Bc, is_s3, R)[meta["features1"]].to_numpy(np.float32))
                qa.append(P.a.to_numpy()); cc.append(code[b]); cn.append(Bc.name_sk.to_numpy()[b]); ca.append(Bc.addr_sk.to_numpy()[b])
                s3.append(np.full(len(b), is_s3, np.int8)); del P, b
            if not Xs:
                pd.DataFrame({"s1_row": np.zeros(0, np.int32), "cand": np.zeros(0, np.int64), "p": np.zeros(0, np.float32)}).to_parquet(out)
                continue
            X1 = np.vstack(Xs); qa = np.concatenate(qa); cc = np.concatenate(cc); cn = np.concatenate(cn)
            ca = np.concatenate(ca); s3 = np.concatenate(s3); del Xs
            p1 = b1.predict(X1, num_threads=N_CPU).astype(np.float32)
            p = p1.copy(); k = p1 >= KEEP
            if k.any():
                S2F = stage2_in_chunks(qa[k], p1[k], cn[k], ca[k], s3[k])
                X2 = np.hstack([X1[k], S2F.to_numpy(np.float32)]); del S2F
                p[k] = np.mean([predict_model(kind, m, X2) for _, kind, m in MODELS], axis=0).astype(np.float32)
                del X2
            pd.DataFrame({"s1_row": A_rowid[rows[qa]], "cand": cc, "p": p}).to_parquet(out, compression="zstd")
            log(country, f"batch {i // BATCH + 1}/{-(-len(a_rows) // BATCH)}", "pairs", len(p), "kept for stage 2", int(k.sum()),
                "| elapsed min", round((time.time() - T0) / 60, 1))
            del X1, qa, cc, cn, ca, s3, p1, p, k, Ac; gc.collect()
        del pools; free_mem()
    log("prediction finished | minutes", round((time.time() - T0) / 60, 1))


# ======================================================================
# 8. COMBINE + France check + write both TSV files
# ======================================================================
if 'combine' in STAGES:
    # ---------------------------------------------------------------------
    # COMBINE: apply the per-country cut-offs, write both TSV files, and print a France check.
    # This cell is cheap (minutes). To change FRANCE_MARGIN, edit it in the settings cell and re-run only this cell.
    # ---------------------------------------------------------------------
    T0 = time.time()
    restore_models()
    meta = json.load(open(f"{WORK}/v10_meta.json"))
    FILES = part_files()          # this session's batches + those of earlier sessions attached as inputs
    log("prediction batch files found:", len(FILES))
    assert FILES, "no prediction batches found: run the predict step first"
    S1t = load(ROOT, "test_source1"); s1_ids = S1t.entity_id.to_numpy(); s1_country = S1t.country.to_numpy(); del S1t
    countries = sorted(set(s1_country))
    if not SMOKE:      # refuse to write a file with empty rows for countries that were never predicted
        have = {os.path.basename(f) for f in FILES}
        missing = [f"{c}_{i}.parquet" for c in countries for i in range(0, int((s1_country == c).sum()), BATCH)
                   if f"{c}_{i}.parquet" not in have]
        assert not missing, f"{len(missing)} prediction batches are missing (e.g. {missing[:3]}): finish the predict step for all countries first"
    cut = {}
    for c in countries:
        if c in meta["cuts"]:
            cut[c] = (meta["cuts"][c]["thr"], meta["cuts"][c]["rel"])
        else:                                              # unseen country (France): average cut-off + margin
            cut[c] = (min(meta["general"]["thr"] + FRANCE_MARGIN, 0.97), meta["general"]["rel"])
        log(f"cut-off for {c}: {cut[c][0]:.3f}, close-to-best share {cut[c][1]:.2f}" + ("" if c in meta["cuts"] else "  (no check data: general rule)"))
    thr_df = pl.DataFrame({"s1_row": np.arange(len(s1_ids), dtype=np.int32),
                           "thr": np.array([cut[c][0] for c in s1_country], np.float32),
                           "rel": np.array([cut[c][1] for c in s1_country], np.float32)})
    lo = float(min(0.30, min(t for t, _ in cut.values())))
    # 1) every pair with p >= lo; each Source 2/3 record goes to its most likely business only
    D0 = (pl.scan_parquet(FILES).filter(pl.col("p") >= lo).collect()
          .sort("p", descending=True).unique(subset="cand", keep="first", maintain_order=True))
    D0 = D0.join(D0.group_by("s1_row").agg(pl.col("p").max().alias("pmax")), on="s1_row").join(thr_df, on="s1_row")
    log("pairs kept for the decision:", D0.height)
    # 2) France check (no labels needed): the share of businesses that get at least one match, and matches per business,
    #    at the same probability level. On check data ~94% of businesses had a match and the average was about 3.5.
    rows_of = {c: pl.Series(np.flatnonzero(s1_country == c).astype(np.int32)) for c in countries}
    print("\nshare of businesses with >= 1 match / average matches per matched business, by probability cut-off")
    print(f"{'cut-off':>8} " + " ".join(f"{c:>16}" for c in countries))
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        row = []
        for c in countries:
            d = D0.filter((pl.col("p") >= t) & pl.col("s1_row").is_in(rows_of[c]))
            n_with = d["s1_row"].n_unique()
            row.append(f"{n_with / rows_of[c].len():6.1%} / {d.height / max(n_with, 1):4.2f}")
        print(f"{t:8.2f} " + " ".join(f"{r:>16}" for r in row))
    print("If France shows a clearly HIGHER share / average than India and US at the same cut-off, it is probably matching too much: "
          "raise FRANCE_MARGIN (try 0.05-0.10). If clearly LOWER, the model is unsure about France: leave 0 or lower it.\n")
    # 3) final decision
    best = D0.filter((pl.col("p") >= pl.col("thr")) & (pl.col("p") >= pl.col("rel") * pl.col("pmax")))
    match = best.group_by("s1_row").agg(decode_expr("cand").sort().alias("ids"))
    mline = np.full(len(s1_ids), "", dtype=object); nmatch = np.zeros(len(s1_ids), np.int32)
    for r, c in zip(match["s1_row"].to_list(), match["ids"].to_list()):
        mline[r] = ",".join(c); nmatch[r] = len(c)
    os.makedirs(f"{WORK}/output", exist_ok=True)
    with open(f"{WORK}/output/matching_results.tsv", "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s, m in zip(s1_ids, mline):
            fh.write(s + "\t" + m + "\n")
    for c in countries:
        rr = s1_country == c
        log(f"  {c}: {int(rr.sum())} businesses, with a match {(nmatch[rr] > 0).mean():.1%}, avg matches {nmatch[rr].mean():.2f}")
    log("done:", len(s1_ids), "rows,", int((nmatch > 0).sum()), "with matches,", int((nmatch == 0).sum()), "empty")
    del match, mline; gc.collect()
    # 4) candidate list: the model's own best-ranked candidates per business, not the full shortlist.
    # Rule update from Amazon (27 Sep): candidate_pairs.tsv counts toward the final ranking, and a SMALLER
    # candidate set per business is ranked higher. KTOP=15 is a generous safety margin (avg true matches ~3.5);
    # every match is still guaranteed to be listed (p >= thr is a necessary condition for being a match).
    KTOP = int(os.environ.get("KTOP", 15))
    D0 = D0.with_columns(pl.col("p").rank(method="ordinal", descending=True).over("s1_row").alias("rk"))
    cand_keep = D0.filter((pl.col("rk") <= KTOP) | (pl.col("p") >= pl.col("thr"))).select(["s1_row", "cand"])
    cgrp = cand_keep.group_by("s1_row").agg(decode_expr("cand").sort().alias("cids"))
    cline = np.full(len(s1_ids), "", dtype=object)
    for r, c in zip(cgrp["s1_row"].to_list(), cgrp["cids"].to_list()):
        cline[r] = ",".join(c)
    with open(f"{WORK}/output/candidate_pairs.tsv", "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s, c in zip(s1_ids, cline):
            fh.write(s + "\t" + c + "\n")
    log(f"candidate file written: avg {cand_keep.height / len(s1_ids):.1f} candidates/business "
        f"(was ~92.6 before this fix) | minutes", round((time.time() - T0) / 60, 1))
    del D0, best, cand_keep, cgrp, cline; gc.collect()


# ======================================================================
# 9. Official format check
# ======================================================================
if 'check' in STAGES:
    # the candidate file is skipped here: checking it needs a lot of memory (the file is still written)
    validator = glob.glob(f'{ROOT}/**/validate_submission.py', recursive=True)[0]
    test_dir = os.path.dirname(find_file(ROOT, 'test_source1')[0])
    subprocess.call([sys.executable, validator, "--matching", f"{WORK}/output/matching_results.tsv", "--test-dir", test_dir, "--candidate", "none.tsv"])   # candidate file check skipped: needs a lot of memory
