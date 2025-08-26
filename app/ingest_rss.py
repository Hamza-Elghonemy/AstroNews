import json, feedparser, hashlib
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtp
from pathlib import Path
from langdetect import detect, LangDetectException



FEEDS = [
    "https://www.jpl.nasa.gov/feeds/news/",
    "https://earthobservatory.nasa.gov/feeds/rss/all.rss",
    "https://www.nasa.gov/news-release/feed/"
]

def is_english(entry):
    """Check if the entry is in English."""
    if entry.get("title") or entry.get("summary"):
        text = entry["title"] + " " + entry["summary"]
        try:
            lang = detect(text)
            return lang == "en"
        except LangDetectException:
            return False
    return False

def normalize(entry, feed_url):

    iso = entry.get("published") or entry.get("updated") or None # Get the published or updated date

    try:
        published_at = dtp.parse(iso).astimezone(timezone.utc).isoformat() # Convert to UTC
    except Exception:
        published_at = datetime.now(timezone.utc).isoformat() # Fallback to current time in UTC


    title = (entry.get("title") or "").strip()
    summary = (entry.get("summary") or entry.get("description") or "").strip()
    url = entry.get("link")
    source = feed_url
    doc_id = hashlib.sha1((url or title).encode()).hexdigest()
    
    
    return {
        "id": doc_id,
        "type": "news",
        "title": title,
        "summary": summary,
        "published_at": published_at,
        "url": url,
        "source": source,
        "topics": []
    }

def run_ingest_rss(max_items_per_feed = 100, since_days=30):
    
    output = []
    
    for feed in FEEDS:
        parsed = feedparser.parse(feed)
        entries = parsed.entries[:max_items_per_feed]
        for entry in entries:
            if is_english(entry):
                item = normalize(entry, feed)
                if item["title"] and item["url"]:
                    output.append(item)
        if since_days is not None:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=since_days)
            output = [item for item in output if dtp.parse(item["published_at"]) >= cutoff_date]

    de_duplicated = {}
    for result in output:
        de_duplicated[result["url"]] = result
        
    output = list(de_duplicated.values())

    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"data/raw/{date_tag}.jsonl")

    with out_path.open("w", encoding="utf-8") as f:
        for result in output:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Finished processing RSS feeds. {len(output)} items written to {out_path}")

if __name__ == "__main__":
    run_ingest_rss()