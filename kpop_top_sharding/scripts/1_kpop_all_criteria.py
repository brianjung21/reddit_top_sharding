#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reddit_brand_crawl_kpop.py

Harvest Reddit posts by brand and aliases across the hot feed only.

For each matched submission, save rich metadata + up to 5 top-level comments
(sorted by 'best'), and record which fields matched the alias (title/selftext/comments).

Outputs:
- ../data/reddit_brand_harvest_hot.csv

Requirements:
  pip install praw pandas tqdm python-dateutil requests

Set your Reddit API creds below or via environment variables:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
"""

from __future__ import annotations
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dateutil import tz
from tqdm import tqdm
import praw
import prawcore
import requests
from requests.exceptions import RequestException
import json
from praw.models import MoreComments

# --------------------------
# Config
# --------------------------
KW_CSV = Path("../files/kpop_keywords_with_aliases.csv")
OUT_DIR = Path("../data")

# Sector tag for notifications
SECTOR = "kpop"

# Crawl knobs
CUTOFF_DAYS = None               # None = no age cutoff; fetch as much as the API will return
MAX_RESULTS_PER_QUERY = 1000     # Reddit search hard-ish ceiling
MAX_COMMENTS = 5                 # top-level comments captured per post
SLEEP_BETWEEN_QUERIES = 1.5      # be gentler between alias×mode queries
SUBMISSION_PAUSE = 0.2        # tiny pause per processed submission
COMMENT_SORT = "best"            # 'best' for comments

FLUSH_EVERY_ROWS = 400               # flush buffers to disk every N rows per mode
MAX_RETRY_PER_ALIASMODE = 10         # cap retries for an alias×mode iterator
CHECKPOINT_PATH = OUT_DIR / "progress_checkpoint.json"
PART_FILES = {
    "hot": OUT_DIR / "reddit_brand_harvest_hot.part.csv",
}
FINAL_FILES = {
    "hot": OUT_DIR / "reddit_brand_harvest_hot.csv",
}

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

def notify_pushover(message: str, title: str = None):  # sector-aware notifier
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

def _sleep_on_429(e: Exception, default_secs: int = 60):
    try:
        retry_after = int(getattr(getattr(e, 'response', None), 'headers', {}).get('retry-after', default_secs))
    except Exception:
        retry_after = default_secs
    print(f"[429] Too many requests. Sleeping {retry_after} seconds…")
    time.sleep(retry_after)

def _sleep_on_neterr(e: Exception, default_secs: int = 60):
    print(f"[neterr] Network error occurred: {e}. Sleeping {default_secs} seconds…")
    time.sleep(default_secs)

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

_buffers = {"hot": []}

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

def build_reddit():
    return praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        ratelimit_seconds=5,
        requestor_kwargs={"timeout": 30},
    )

def compile_alias_patterns(aliases: List[str]) -> List[re.Pattern]:
    """
    STRICT alias regex rules:
    - Latin single-word aliases: contiguous whole-word (no punctuation inside), e.g. \bIVE\b
    - Latin multi-word aliases: allow **whitespace only** between tokens (no punctuation),
      e.g. \bNEW\s+JEANS\b (not NEW-JEANS, NEW.JEANS, or I,VE)
    - Non-Latin (e.g., Hangul): match literal substring (no word-boundary semantics),
      using re.escape without adding separators.
    """
    pats: List[re.Pattern] = []
    for raw in aliases:
        a = (raw or "").strip()
        if not a:
            continue
        # Detect any Latin letter in alias
        has_latin = re.search(r"[A-Za-z]", a) is not None
        if has_latin:
            # Split on whitespace for multi-token aliases
            tokens = a.split()
            if len(tokens) == 1:
                # Contiguous whole word; forbid punctuation inside
                patt = rf"(?<![A-Za-z0-9_]){re.escape(tokens[0])}(?![A-Za-z0-9_])"
            else:
                # Allow **whitespace only** between tokens (no punctuation separators)
                joined = r"\s+".join(re.escape(t) for t in tokens)
                patt = rf"(?<![A-Za-z0-9_]){joined}(?![A-Za-z0-9_])"
            pats.append(re.compile(patt, re.IGNORECASE))
        else:
            # Non-Latin (e.g., Hangul) — match literal substring
            pats.append(re.compile(re.escape(a)))
    return pats

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
            _sleep_on_neterr(e, default_secs=10)
            continue
    return comments_text

# NOTE: Matching occurs on raw title/selftext/comments (no pre-clean). This preserves punctuation
# like apostrophes so that patterns such as \bIVE\b will NOT match "I've" or "I,VE".
def submission_row(brand: str, alias_used: str, mode: str, pats: List[re.Pattern], s) -> Dict:
    try:
        title = _u(getattr(s, "title", ""))
        selftext = _u(getattr(s, "selftext", ""))
        # First pass: check explicit match in title/selftext only
        explicit, matched_fields = find_match_fields(title, selftext, [], pats)
        comments: List[str] = []
        if not explicit and getattr(s, 'num_comments', 0):
            comments = first_n_comments_best(s, MAX_COMMENTS)
            explicit, matched_fields = find_match_fields(title, selftext, comments, pats)
        if not explicit:
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
            "mode": mode,
            "post_id": getattr(s, "id", None),
            "permalink": f"https://www.reddit.com{getattr(s, 'permalink', '')}",
            "url": getattr(s, "url", None),
            "title": title,
            "selftext": selftext,
            "comments_text": " || ".join(_u(x) for x in comments),
            "matched_fields": matched_fields,
            "subreddit": subreddit_name,
            "author": author_name,
            "score": getattr(s, "score", None),
            "ups": getattr(s, "ups", None),
            "upvote_ratio": getattr(s, "upvote_ratio", None),
            "num_comments": getattr(s, "num_comments", None),
            "over_18": getattr(s, "over_18", None),
            "stickied": getattr(s, "stickied", None),
            "is_self": getattr(s, "is_self", None),
            "locked": getattr(s, "locked", None),
            "removed_by_category": getattr(s, "removed_by_category", None),
            "distinguished": getattr(s, "distinguished", None),
            "link_flair_text": getattr(s, "link_flair_text", None),
            "domain": getattr(s, "domain", None),
            # no crosspost_parent to avoid lazy fetch
            "created_utc": created_ts,
            "created_iso": created_iso,
        }
        return row
    except prawcore.exceptions.TooManyRequests as e:
        _sleep_on_429(e)
        return {}
    except (prawcore.exceptions.RequestException, Exception) as e:
        _sleep_on_neterr(e, default_secs=10)
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

def cutoff_dt_utc(days: Optional[int]) -> datetime:
    if days is None:
        # effectively no cutoff; return the earliest possible aware datetime
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - timedelta(days=days)

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

def crawl_brand_alias_mode(reddit, brand: str, alias: str, mode: str, pats: List[re.Pattern]):
    """mode: 'new' | 'hot' | 'top:month'"""
    if mode == "top:month":
        sort = "top"; tf = "month"
    elif mode == "hot":
        sort = "hot"; tf = None
    else:
        sort = "new"; tf = None
    q = f"\"{alias}\""
    cutoff = cutoff_dt_utc(CUTOFF_DAYS)
    retries = 0
    while True:
        try:
            itr = subreddit_all_search(reddit, q, sort=sort, time_filter=tf, limit=MAX_RESULTS_PER_QUERY)
            for s in itr:
                if CUTOFF_DAYS is not None and sort in ("new", "hot"):
                    try:
                        created = datetime.fromtimestamp(getattr(s, "created_utc", 0), tz=timezone.utc)
                        if created < cutoff:
                            raise StopIteration
                    except Exception:
                        pass
                row = submission_row(brand, alias, mode, pats, s)
                if row:
                    _buffers[mode].append(row)
                    if len(_buffers[mode]) >= FLUSH_EVERY_ROWS:
                        flush_mode_buffer(mode)
                time.sleep(SUBMISSION_PAUSE)
            break
        except prawcore.exceptions.TooManyRequests as e:
            _sleep_on_429(e)
            retries += 1
            if retries >= MAX_RETRY_PER_ALIASMODE:
                print(f"[retry] giving up on {brand}/{alias}/{mode} after {retries} retries (429)")
                break
            continue
        except prawcore.exceptions.RequestException as e:
            _sleep_on_neterr(e, default_secs=15)
            retries += 1
            if retries >= MAX_RETRY_PER_ALIASMODE:
                print(f"[retry] giving up on {brand}/{alias}/{mode} after {retries} retries (net)")
                break
            # recreate iterator and continue
            continue
        except Exception as e:
            _sleep_on_neterr(e, default_secs=10)
            retries += 1
            if retries >= MAX_RETRY_PER_ALIASMODE:
                print(f"[retry] giving up on {brand}/{alias}/{mode} after {retries} retries (unknown)")
                break
            continue
    time.sleep(SLEEP_BETWEEN_QUERIES)

def run():
    reddit = build_reddit()
    t0 = time.time()
    brand2aliases = load_brand_aliases(KW_CSV)
    modes = ["hot"]
    done = load_checkpoint()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done0 = len(done)
    cutoff_label = CUTOFF_DAYS if CUTOFF_DAYS is not None else "ALL"
    try:
        notify_pushover(
            message=(
                f"[{SECTOR}] Harvest STARTED at {start_ts}\n"
                f"brands={len(brand2aliases)}; modes={','.join(modes)}; cutoff_days={cutoff_label}\n"
                f"already_done={done0} alias×mode combos (from checkpoint)"
            ),
            title=f"[{SECTOR}] Reddit harvest started",
        )
    except Exception:
        pass
    print(f"[info] Brands to crawl: {len(brand2aliases)}; modes={modes}; cutoff_days={cutoff_label}")
    for brand, aliases in tqdm(brand2aliases.items(), desc="Brands"):
        pats = compile_alias_patterns(aliases)
        for alias in aliases:
            for mode in modes:
                key = (brand, alias, mode)
                if key in done:
                    continue
                crawl_brand_alias_mode(reddit, brand, alias, mode, pats)
                flush_mode_buffer(mode)
                done.add(key)
                save_checkpoint(done)
    # Flush any remaining buffers
    for m in ["hot"]:
        flush_mode_buffer(m)
    # Finalize: aggregate part files -> final CSVs
    counts = {}
    total_rows = 0
    for m in ["hot"]:
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
    elapsed = time.time() - t0
    hh = int(elapsed // 3600); mm = int((elapsed % 3600) // 60); ss = int(elapsed % 60)
    try:
        notify_pushover(
            message=(
                f"[{SECTOR}] Harvest FINISHED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"elapsed={hh:02d}:{mm:02d}:{ss:02d}\n"
                f"processed_alias_modes={processed_alias_modes}; total_done={len(done)}\n"
                f"rows_total={total_rows} | hot={counts.get('hot', 0)}"
            ),
            title=f"[{SECTOR}] Reddit harvest completed",
        )
    except Exception:
        pass

if __name__ == "__main__":
    t0 = None
    try:
        # If run() sets t0, we want to access it in excepts
        run()
    except KeyboardInterrupt:
        try:
            for m in ["hot"]:
                flush_mode_buffer(m)
        except Exception:
            pass
        try:
            # Compute elapsed safely
            try:
                elapsed = time.time() - t0 if t0 is not None else 0
                hh = int(elapsed // 3600); mm = int((elapsed % 3600) // 60); ss = int(elapsed % 60)
            except Exception:
                hh = mm = ss = 0
            notify_pushover(
                message=f"[{SECTOR}] Harvest INTERRUPTED (Ctrl+C) after {hh:02d}:{mm:02d}:{ss:02d}",
                title=f"[{SECTOR}] Reddit harvest interrupted",
            )
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            for m in ["hot"]:
                flush_mode_buffer(m)
        except Exception:
            pass
        try:
            try:
                elapsed = time.time() - t0 if t0 is not None else 0
                hh = int(elapsed // 3600); mm = int((elapsed % 3600) // 60); ss = int(elapsed % 60)
            except Exception:
                hh = mm = ss = 0
            notify_pushover(
                message=f"[{SECTOR}] Harvest FAILED after {hh:02d}:{mm:02d}:{ss:02d}: {type(e).__name__}: {e}",
                title=f"[{SECTOR}] Reddit harvest FAILED",
            )
        except Exception:
            pass
        raise
