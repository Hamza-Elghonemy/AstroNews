import json, re
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from math import exp
from typing import List, Dict
from text_utils import tokenize, contains_word

def latest_jsonl_files(directory = "data/raw") -> List[Path]:
    """Returns a list of the latest JSONL files in the given directory.
    Args:
        directory (str): The directory to search for JSONL files.
    Returns:
        List[Path]: A list of the latest JSONL files.
    """
    
    jsonl_files = sorted(Path(directory).glob("*.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError("No JSONL files found")
    
    return jsonl_files[-1]

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load a JSONL file and return a list of dictionaries.
    Args:
        file_path (Path): The path to the JSONL file.
    Returns:
        List[Dict]: A list of dictionaries representing the JSON objects in the file.
    """
    
    items = []
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def as_utc(ts: str) -> datetime:
    """Convert a timestamp string to a UTC datetime.

    Args:
        ts (str): The timestamp string to convert.

    Returns:
        datetime: The corresponding UTC datetime.
    """
    if not ts or not isinstance(ts, str):
        # Very old fallback (effectively recency ~ 0)
        return datetime.now(timezone.utc).replace(year=2000)

    s = ts.strip()
    try:
        # Normalize trailing Z to +00:00
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc).replace(year=2000)

def textify(item: dict) -> str:
    """Convert a news article item to a plain text representation.

    Args:
        item (dict): The news article item.

    Returns:
        str: The plain text representation of the news article.
    """
    title = item.get("title") or ""
    summary = item.get("summary") or ""

    return f"{title}\n\n{summary}".strip()
    
# --------------------------------------------------------
def recency_boost(published_at, now: datetime, tau_days: float=14.0) -> float:
    """
    Apply a recency boost to the relevance score based on the publication date.
    Args:
        published_at (datetime): The publication date as a datetime object.
        now (datetime): The current date and time as a datetime object.
        tau_days (float): The time constant for the decay function, in days.
    Returns:
        float: The boosted relevance score.
    """
    if isinstance(published_at, str):
        dt = as_utc(published_at)
    elif isinstance(published_at, datetime):
        dt = published_at.astimezone(timezone.utc) if published_at.tzinfo else published_at.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.now(timezone.utc).replace(year=2000)

    age_days = max(0.0, (now - dt).total_seconds() / 86400.0)  # convert seconds → days
    return exp(-age_days / tau_days)

def keyword_score(query_tokens: List[str], title: str, summary: str) -> float:
    """
    Calculate the keyword score for a given text based on the query tokens.
    Args:
        query_tokens (List[str]): The list of query tokens.
        title (str): The title to evaluate.
        summary (str): The summary to evaluate.
    Returns:
        float: The keyword score.
    """
    
    t_title = tokenize(title)
    t_sum = tokenize(summary)
    c_title = Counter(t_title)
    c_sum = Counter(t_sum)
    
    title_hits = sum(c_title[token] for token in query_tokens)
    sum_hits = sum(c_sum[token] for token in query_tokens)
    
    # weights --> title 3x summary
    title_score = title_hits * 3
    summary_score = sum_hits

    return title_score + summary_score

def score_item(item: dict, query_tokens: str, now: datetime) -> float:
    """Score a news article item based on the query tokens and publication date.

    Args:
        item (dict): The news article item.
        query_tokens (str): The query tokens.
        now (datetime): The current date and time.

    Returns:
        float: The overall score for the news article item.
    """
    ts = item.get("published_at")
    published_dt = as_utc(ts) if ts else now.replace(year=now.year - 10)
    
    title = item.get("title") or ""
    summary = item.get("summary") or ""
    q_tokens = tokenize(query_tokens)

    kw = keyword_score(q_tokens, title, summary)
    exact_bonus = 0.0
    for tok in set(q_tokens):
        if contains_word(title, tok):
            exact_bonus += 2.0
        elif contains_word(summary, tok):
            exact_bonus += 0.5
            
    # If no keyword evidence at all, downweight heavily (so recency doesn't dominate)        
    if kw == 0 and exact_bonus == 0:
        kw_penalized = 0.1
    else:
        kw_penalized = kw + exact_bonus 
    
    recency = recency_boost(published_dt, now)
    
    return 0.85 * kw_penalized + 0.15 * recency 

def search(items: List[Dict], query: str, k: int = 10) -> List[Dict]:
    now = datetime.now(timezone.utc)
    scored = []
    for item in items:
        s = score_item(item, query, now)
        scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]

# -----------DEMO-----------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Local keyword+recency search over RSS JSONL")
    ap.add_argument("query", type=str, help="Your search query, e.g., 'Artemis' or 'Dragon'")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--path", type=str, default="data/raw")
    args = ap.parse_args()

    latest = latest_jsonl_files(args.path)
    items = load_jsonl(latest)
    hits = search(items, args.query, k=args.k)

    print(f"\nTop results for: {args.query!r} (from {latest.name})\n")
    for i, item in enumerate(hits, 1):
        date = (item.get("published_at") or item.get("published") or "")[:10]
        print(f"[{i}] {date} — {item.get('title')}")
        print(f"     {item.get('url')}")