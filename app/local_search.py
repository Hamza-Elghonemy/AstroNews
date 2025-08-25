import json, re
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from math import exp
from typing import List, Dict




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

def as_utc(timestamp: float) -> datetime:
    """Convert a Unix timestamp to a UTC datetime.
    Args:
        timestamp (float): The Unix timestamp to convert.
    Returns:
        datetime: The corresponding UTC datetime.
    """
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc) # 
    except (OSError, ValueError):
        return datetime.fromisoformat(timestamp.strip("+")[0]).replace(tzinfo=timezone.utc) 

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
def recency_boost(published_at: datetime, now: datetime, tau_days: float=14.0) -> float:
    """
    Apply a recency boost to the relevance score based on the publication date.
    Args:
        published_at (datetime): The publication date as a datetime object.
        now (datetime): The current date and time as a datetime object.
        tau_days (float): The time constant for the decay function, in days.
    Returns:
        float: The boosted relevance score.
    """
    day = max(0.0, (now - published_at).days - tau_days)
    return exp(-day / tau_days)

def keyword_score(query_tokens: List[str], text: str) -> float:
    """
    Calculate the keyword score for a given text based on the query tokens.
    Args:
        query_tokens (List[str]): The list of query tokens.
        text (str): The text to evaluate.
    Returns:
        float: The keyword score.
    """
    text_tokens = re.findall(r"[a-z0-9]+", text.lower())
    count = Counter(text_tokens)
    return sum(count[token] for token in query_tokens) / len(query_tokens) if query_tokens else 0.0 # Avoid division by zero

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
    if not ts:
        published_dt = now.replace(year=now.year-10)
    else:
        published_dt = as_utc(ts)
        
    recency = recency_boost(published_dt, now)
    keywords = keyword_score(query_tokens, textify(item))
    
    return recency * 0.3 + 0.7 * keywords 

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