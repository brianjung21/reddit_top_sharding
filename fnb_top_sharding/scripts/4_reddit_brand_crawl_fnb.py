#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_reddit_brand_crawl_kpop.py

Harvest Reddit posts by brand and aliases across three feeds:
- new   (chronological, stops at cutoff_days)
- hot   (practical proxy for "best")
- top:month (top posts in the past month)

For each matched submission, save rich metadata + up to 5 top-level comments
(sorted by 'best'), and record which fields matched the alias (title/selftext/comments).

Outputs:
- ../data/reddit_brand_harvest_top_month.csv

Requirements:
  pip install praw pandas tqdm python-dateutil requests
  langdetect (optional)

Set your Reddit API creds below or via environment variables:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
"""

from __future__ import annotations
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import prawcore
import requests
from typing import Optional

import pandas as pd
from dateutil import tz
from tqdm import tqdm
import praw
import json

from praw.models import MoreComments

# Language detection is optional; we fall back to heuristics if not installed
try:
    from langdetect import detect as _ld_detect
except Exception:
    _ld_detect = None

# --------------------------
# Config
# --------------------------
KW_CSV = Path("../files/fnb_keywords_with_aliases.csv")

OUT_DIR = Path("../data")
SECTOR = "fnb"

# Crawl knobs
CUTOFF_DAYS = 30                 # for new/hot, stop when results are older than this many days
MAX_RESULTS_PER_QUERY = 1000     # Reddit search hard-ish ceiling
MAX_COMMENTS = 5                 # top-level comments captured per post
SLEEP_BETWEEN_QUERIES = 1.5      # gentler throttle per alias-mode query
SUBMISSION_PAUSE = 0.2           # tiny pause per processed submission
COMMENT_SORT = "best"            # 'best' for comments

# Robustness knobs
FLUSH_EVERY_ROWS = 400               # flush buffers to disk every N rows per mode
MAX_RETRY_PER_ALIASMODE = 10         # cap retries for an alias×mode iterator
CHECKPOINT_PATH = OUT_DIR / "progress_checkpoint.json"
PART_FILES = {
    "top:month": OUT_DIR / "reddit_brand_harvest_top_month.part.csv",
}
FINAL_FILES = {
    "top:month": OUT_DIR / "reddit_brand_harvest_top_month.csv",
}

# --- Filtering knobs (Steps 3 & 4) ---
ENABLE_LANGUAGE_FILTER = True    # keep English-only
ENABLE_CUE_FILTER = True         # require sector cue words somewhere in the text
CUE_TERMS_PATH = Path("../data/sector_terms.csv")  # expects columns: term, frequency, tfidf_score
CUE_MIN_FREQ = 5                 # drop ultra-rare terms from cue list
CUE_TOP_TFIDF_PCT = 0.10         # keep top 10% by tf-idf (if column present)
CUE_MAX_TERMS = 2000             # safety cap on number of cue terms compiled

# Reddit credentials (directly defined)
CLIENT_ID = ""
CLIENT_SECRET = ""
USER_AGENT = ""

# Pushover credentials
PUSHOVER_USER_KEY = ""
PUSHOVER_API_TOKEN = ""

# --------------------------
# Helpers
# --------------------------

def build_reddit():
    return praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        ratelimit_seconds=5,
        requestor_kwargs={"timeout": 30},  # increase HTTP read/connect timeout
    )

def _sleep_on_429(e: Exception, default_secs: int = 60):
    try:
        retry_after = int(getattr(getattr(e, 'response', None), 'headers', {}).get('retry-after', default_secs))
    except Exception:
        retry_after = default_secs
    print(f"[429] Too many requests. Sleeping {retry_after} seconds…")
    time.sleep(retry_after)

def _sleep_on_neterr(e: Exception, secs: int = 10):
    print(f"[net] Network exception: {type(e).__name__}: {e}. Sleeping {secs}s…")
    time.sleep(secs)

def _u(s: str) -> str:
    if s is None:
        return ""
    try:
        return str(s)
    except Exception:
        try:
            return s.encode("utf-8", "replace").decode("utf-8", "replace")
        except Exception:
            return ""

# --- Language and cue helpers ---
HANGUL_RE = re.compile(r"[\u3130-\u318F\uAC00-\uD7AF]")
LATIN_RE  = re.compile(r"[A-Za-z]")

def looks_english(text: str) -> bool:
    """Return True if text appears to be English.
    Strategy: reject if Hangul present; else if langdetect available, require 'en'; otherwise require some Latin letters.
    """
    t = (text or "")
    if HANGUL_RE.search(t):
        return False
    if _ld_detect is not None:
        try:
            lang = _ld_detect(t)
            return lang == "en"
        except Exception:
            pass
    return bool(LATIN_RE.search(t))

def load_checkpoint() -> set:
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(tuple(x) for x in data.get("done_alias_modes", []))
    except Exception:
        return set()

def save_checkpoint(done_set: set):
    try:
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump({"done_alias_modes": [list(x) for x in sorted(done_set)]}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[ckpt] failed to save checkpoint: {e}")

_buffers = {"top:month": []}

def flush_mode_buffer(mode: str):
    buf = _buffers.get(mode, [])
    if not buf:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = PART_FILES[mode]
    df_part = pd.DataFrame(buf)
    header = not path.exists()
    try:
        df_part.to_csv(
            path,
            mode="a",
            header=header,
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as e:
        print(f"[flush] failed to write {path}: {e}")
        return
    _buffers[mode].clear()

def compile_alias_patterns(aliases: List[str]) -> List[re.Pattern]:
    """
    Strict alias regex: word boundaries, allow separators between tokens (space/_/.-),
    case-insensitive.
    """
    pats = []
    for a in aliases:
        a = a.strip()
        if not a:
            continue
        toks = [re.escape(t) for t in a.split()]
        sep = r"[-_\.\s]*"
        patt = rf"(?<!\w){sep.join(toks)}(?!\w)"
        pats.append(re.compile(patt, re.IGNORECASE))
    return pats

# --- Cue terms loading and pattern helpers ---
def compile_phrase_pattern(phrase: str) -> re.Pattern:
    """Compile a whole-word-ish regex for a unigram/bigram phrase (Latin only).
    Non-letter characters are treated as separators; case-insensitive.
    """
    phrase = phrase.strip().lower()
    if not phrase:
        return re.compile(r"$^")
    toks = [re.escape(t) for t in re.split(r"\s+", phrase) if t]
    sep = r"[-_\.\s]+"
    patt = rf"(?<!\w){sep.join(toks)}(?!\w)"
    return re.compile(patt, re.IGNORECASE)

def load_cue_terms(path: Path) -> List[str]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    cols = {c.lower(): c for c in df.columns}
    term_col = cols.get("term") or cols.get("word") or cols.get("token")
    if not term_col:
        return []
    # Optional filters
    if "tfidf_score" in df.columns:
        cutoff = df["tfidf_score"].quantile(1 - CUE_TOP_TFIDF_PCT)
        df = df[df["tfidf_score"] >= cutoff]
    if "frequency" in df.columns:
        df = df[df["frequency"] >= CUE_MIN_FREQ]
    terms = (
        df[term_col]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )
    # Cap and dedupe
    out = []
    seen = set()
    for t in terms:
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= CUE_MAX_TERMS:
            break
    return out

def compile_cue_patterns(terms: List[str]) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    for t in terms:
        # ignore single-character terms
        if len(t) < 2:
            continue
        pats.append(compile_phrase_pattern(t))
    return pats

def hits_cue(text: str, cue_pats: List[re.Pattern]) -> bool:
    if not cue_pats:
        return True  # if no cues loaded, do not block
    t = (text or "")
    for p in cue_pats:
        if p.search(t):
            return True
    return False

def hits_alias(text: str, pats: List[re.Pattern]) -> bool:
    t = (text or "")
    for p in pats:
        if p.search(t):
            return True
    return False

def find_match_fields(title: str, selftext: str, comments: List[str], pats: List[re.Pattern]) -> Tuple[bool,str]:
    fields = []
    if hits_alias(title, pats):
        fields.append("title")
    if hits_alias(selftext, pats):
        fields.append("selftext")
    # only check comments if needed (we’ll often fetch them anyway for saving)
    if comments and any(hits_alias(c, pats) for c in comments):
        fields.append("comments")
    return (len(fields) > 0, "|".join(fields))

def first_n_comments_best(subm, n: int) -> List[str]:
    comments_text: List[str] = []
    tries = 0
    while tries < 3:
        tries += 1
        try:
            try:
                subm.comment_sort = COMMENT_SORT
            except Exception:
                pass
            # Pass 1: first page only (no replace_more)
            count = 0
            for c in subm.comments:
                if isinstance(c, MoreComments):
                    continue
                body = _u(getattr(c, "body", ""))
                lo = body.strip().lower()
                if not body or lo in ("[deleted]", "[removed]"):
                    continue
                comments_text.append(body.strip())
                count += 1
                if count >= n:
                    break
            if len(comments_text) >= n:
                break
            # Pass 2: one light expansion only
            try:
                subm.comments.replace_more(limit=1)
            except Exception:
                pass
            for c in subm.comments:
                if isinstance(c, MoreComments):
                    continue
                body = _u(getattr(c, "body", ""))
                lo = body.strip().lower()
                if not body or lo in ("[deleted]", "[removed]"):
                    continue
                if len(comments_text) < n:
                    comments_text.append(body.strip())
                else:
                    break
            break
        except prawcore.exceptions.TooManyRequests as e:
            _sleep_on_429(e)
            continue
        except (prawcore.exceptions.RequestException, Exception) as e:
            _sleep_on_neterr(e, secs=10)
            continue
    return comments_text

def submission_row(brand: str, alias_used: str, mode: str, pats: List[re.Pattern], s, cue_pats: List[re.Pattern]) -> Dict:
    """Extract rich metadata + comments on demand; ensure explicit alias mention and apply filters."""
    try:
        title = _u(getattr(s, "title", ""))
        selftext = _u(getattr(s, "selftext", ""))

        # First pass: check explicit match in title/selftext only
        explicit, matched_fields = find_match_fields(title, selftext, [], pats)

        comments = []
        if not explicit and getattr(s, 'num_comments', 0):
            # Second pass: fetch up to N comments (expensive)
            comments = first_n_comments_best(s, MAX_COMMENTS)
            explicit, matched_fields = find_match_fields(title, selftext, comments, pats)

        if not explicit:
            return {}

        # Step 3: English-only filter
        if ENABLE_LANGUAGE_FILTER:
            blob = f"{title}\n{selftext}\n" + ("\n".join(comments) if comments else "")
            if not looks_english(blob):
                return {}

        # Step 4: Sector cue words filter (presence anywhere in blob)
        if ENABLE_CUE_FILTER:
            if not hits_cue(blob, cue_pats):
                return {}

        created_ts = getattr(s, "created_utc", None)
        created_iso = None
        if created_ts is not None:
            try:
                created_iso = datetime.fromtimestamp(created_ts, tz=timezone.utc).isoformat()
            except Exception:
                created_iso = None

        subreddit_name = None
        try:
            subreddit_name = getattr(s.subreddit, "display_name", None)
        except Exception:
            pass

        author_name = None
        try:
            author = getattr(s, "author", None)
            author_name = getattr(author, "name", None)
        except Exception:
            pass

        row = {
            "brand": brand,
            "alias_used": alias_used,
            "mode": mode,                      # 'new' | 'hot' | 'top:month'
            "post_id": getattr(s, "id", None),
            "permalink": f"https://www.reddit.com{getattr(s, 'permalink', '')}",
            "url": getattr(s, "url", None),
            "title": title,
            "selftext": selftext,
            "comments_text": " || ".join(_u(x) for x in comments),
            "matched_fields": matched_fields,

            # Subreddit / author (no extra fetches)
            "subreddit": subreddit_name,
            "author": author_name,

            # Scores
            "score": getattr(s, "score", None),
            "ups": getattr(s, "ups", None),
            "upvote_ratio": getattr(s, "upvote_ratio", None),
            "num_comments": getattr(s, "num_comments", None),

            # Flags
            "over_18": getattr(s, "over_18", None),
            "stickied": getattr(s, "stickied", None),
            "is_self": getattr(s, "is_self", None),
            "locked": getattr(s, "locked", None),
            "removed_by_category": getattr(s, "removed_by_category", None),
            "distinguished": getattr(s, "distinguished", None),

            # Flair / domain
            "link_flair_text": getattr(s, "link_flair_text", None),
            "domain": getattr(s, "domain", None),

            # Timestamps
            "created_utc": created_ts,
            "created_iso": created_iso,
        }
        return row
    except prawcore.exceptions.TooManyRequests as e:
        _sleep_on_429(e)
        return {}
    except (prawcore.exceptions.RequestException, Exception) as e:
        _sleep_on_neterr(e, secs=10)
        return {}

def subreddit_all_search(reddit, query: str, sort: str, time_filter: Optional[str] = None, limit: int = MAX_RESULTS_PER_QUERY):
    """
    Search r/all. Sort in {'new','hot','top','relevance','comments'}.
    For 'top', you can pass time_filter in {'hour','day','week','month','year','all'}.
    """
    sub_all = reddit.subreddit("all")
    kwargs = dict(limit=limit, sort=sort)
    if sort == "top" and time_filter:
        kwargs["time_filter"] = time_filter
    return sub_all.search(query, **kwargs)

def cutoff_dt_utc(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)

def notify_pushover(message: str, title: str = None):
    if title is None:
        title = f"[{SECTOR}] Reddit Harvest"
    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_API_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "title": title,
                "message": message,
                "priority": 0,
            },
            timeout=3,
        )
        if resp.status_code != 200:
            print(f"[pushover] non-200 response: {resp.status_code} {resp.text[:120]}")
    except Exception as e:
        print(f"[pushover] failed to send: {e}")

# --------------------------
# Main crawl
# --------------------------

def load_brand_aliases(path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    if "Keyword" not in df.columns or "Aliases" not in df.columns:
        raise ValueError("Expected columns 'Keyword' and 'Aliases' in the aliases CSV.")
    brand2aliases: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        brand = str(r["Keyword"]).strip().lower()
        aliases = [a.strip() for a in str(r["Aliases"]).split("|") if a.strip()]
        # Include main brand name first, then aliases
        full_list = [brand] + [a for a in aliases if a.lower() != brand]
        brand2aliases[brand] = full_list
    return brand2aliases

def crawl_brand_alias_mode(reddit, brand: str, alias: str, mode: str, pats: List[re.Pattern], cue_pats: List[re.Pattern]):
    """
    mode: 'new' | 'hot' | 'top:month'
    """
    if mode == "top:month":
        sort = "top"; tf = "month"
    elif mode == "hot":
        sort = "hot"; tf = None
    else:
        sort = "new"; tf = None

    q = f"\"{alias}\""   # quote the alias to tighten search
    cutoff = cutoff_dt_utc(CUTOFF_DAYS)
    retries = 0
    while True:
        try:
            itr = subreddit_all_search(reddit, q, sort=sort, time_filter=tf, limit=MAX_RESULTS_PER_QUERY)
            for s in itr:
                # For new/hot, break once results get older than cutoff (ordered feeds).
                if sort in ("new", "hot"):
                    try:
                        created = datetime.fromtimestamp(getattr(s, "created_utc", 0), tz=timezone.utc)
                        if created < cutoff:
                            raise StopIteration
                    except Exception:
                        pass

                row = submission_row(brand, alias, mode, pats, s, cue_pats)
                if row:
                    _buffers[mode].append(row)
                    if len(_buffers[mode]) >= FLUSH_EVERY_ROWS:
                        flush_mode_buffer(mode)

                time.sleep(SUBMISSION_PAUSE)
            break  # exhausted iterator
        except prawcore.exceptions.TooManyRequests as e:
            _sleep_on_429(e)
            retries += 1
            if retries >= MAX_RETRY_PER_ALIASMODE:
                print(f"[retry] giving up on {brand}/{alias}/{mode} after {retries} retries (429)")
                break
            continue
        except prawcore.exceptions.RequestException as e:
            _sleep_on_neterr(e, secs=15)
            retries += 1
            if retries >= MAX_RETRY_PER_ALIASMODE:
                print(f"[retry] giving up on {brand}/{alias}/{mode} after {retries} retries (net)")
                break
            continue
        except Exception as e:
            _sleep_on_neterr(e, secs=10)
            retries += 1
            if retries >= MAX_RETRY_PER_ALIASMODE:
                print(f"[retry] giving up on {brand}/{alias}/{mode} after {retries} retries (unknown)")
                break
            continue

    time.sleep(SLEEP_BETWEEN_QUERIES)

def run():
    reddit = build_reddit()
    t_start = time.time()
    brand2aliases = load_brand_aliases(KW_CSV)
    # Load cue terms and compile patterns, if enabled
    cue_terms = load_cue_terms(CUE_TERMS_PATH) if ENABLE_CUE_FILTER else []
    cue_pats = compile_cue_patterns(cue_terms) if cue_terms else []
    if ENABLE_CUE_FILTER:
        print(f"[cue] loaded {len(cue_terms)} cue terms; compiled {len(cue_pats)} regexes from {CUE_TERMS_PATH}")
    modes = ["top:month"]
    done = load_checkpoint()
    done0 = len(done)

    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Removed pushover notification at the start of the harvest

    print(f"[info] Brands to crawl: {len(brand2aliases)}; modes={modes}; cutoff_days={CUTOFF_DAYS}")
    for brand, aliases in tqdm(brand2aliases.items(), desc="Brands"):
        pats = compile_alias_patterns(aliases)
        for alias in aliases:
            for mode in modes:
                key = (brand, alias, mode)
                if key in done:
                    continue
                crawl_brand_alias_mode(reddit, brand, alias, mode, pats, cue_pats)
                flush_mode_buffer(mode)
                done.add(key)
                save_checkpoint(done)

    # Flush remaining buffers
    for m in ["top:month"]:
        flush_mode_buffer(m)

    # Finalize: aggregate part files -> final CSVs
    counts = {}
    total_rows = 0
    for m in ["top:month"]:
        part = PART_FILES[m]
        if part.exists():
            try:
                df_m = pd.read_csv(part)
                df_m = df_m.drop_duplicates(subset=["brand","post_id","mode","alias_used"]).sort_values(["brand","created_utc","mode"]).reset_index(drop=True)
                FINAL_FILES[m].parent.mkdir(parents=True, exist_ok=True)
                df_m.to_csv(FINAL_FILES[m], index=False, encoding="utf-8-sig")
                counts[m] = len(df_m)
                total_rows += counts[m]
                print(f"[ok] wrote {FINAL_FILES[m]} (rows={counts[m]})")
            except Exception as e:
                print(f"[finalize] failed to aggregate {part}: {e}")
        else:
            counts[m] = 0

    processed_alias_modes = max(0, len(done) - done0)
    elapsed = time.time() - t_start
    hh = int(elapsed // 3600)
    mm = int((elapsed % 3600) // 60)
    ss = int(elapsed % 60)
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notify_pushover(
        message=(
            f"[{SECTOR}] Harvest FINISHED at {end_ts}\n"
            f"elapsed={hh:02d}:{mm:02d}:{ss:02d}\n"
            f"processed_alias_modes={processed_alias_modes}; total_done={len(done)}\n"
            f"rows_total={total_rows} | top:month={counts.get('top:month', 0)}"
        ),
        title=f"[{SECTOR}] Reddit harvest completed",
    )

if __name__ == "__main__":
    t0 = time.time()
    try:
        run()
    except KeyboardInterrupt:
        try:
            for m in ["top:month"]:
                flush_mode_buffer(m)
        except Exception:
            pass
        try:
            elapsed = time.time() - t0
            hh = int(elapsed // 3600); mm = int((elapsed % 3600) // 60); ss = int(elapsed % 60)
        except Exception:
            hh = mm = ss = 0
        try:
            notify_pushover(
                message=f"[{SECTOR}] Harvest INTERRUPTED (Ctrl+C) after {hh:02d}:{mm:02d}:{ss:02d}",
                title=f"[{SECTOR}] Reddit harvest interrupted",
            )
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            for m in ["top:month"]:
                flush_mode_buffer(m)
        except Exception:
            pass
        try:
            elapsed = time.time() - t0
            hh = int(elapsed // 3600); mm = int((elapsed % 3600) // 60); ss = int(elapsed % 60)
        except Exception:
            hh = mm = ss = 0
        try:
            notify_pushover(
                message=f"[{SECTOR}] Harvest FAILED after {hh:02d}:{mm:02d}:{ss:02d}: {type(e).__name__}: {e}",
                title=f"[{SECTOR}] Reddit harvest FAILED",
            )
        except Exception:
            pass
        raise
