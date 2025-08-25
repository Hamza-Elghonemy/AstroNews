from collections import Counter
import re

STOPWORDS = {
    "the","a","an","to","of","and","in","on","for","at","from","by","with","as",
    "is","are","was","were","be","been","being","that","this","these","those",
    "it","its","into","over","about","than","up","down","out","off","or","not"
}
WORD_RE = re.compile(r"[a-z0-9]+")

def tokenize(s: str):
    return [w for w in WORD_RE.findall((s or "").lower()) if w not in STOPWORDS]

def contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word.lower())}\b", (text or "").lower()) is not None

def keyword_hits(query: str, title: str, summary: str = "") -> tuple[int, int]:
    q_tokens = tokenize(query)
    t_title  = tokenize(title)
    t_sum    = tokenize(summary)
    # simple term-frequency counts
    title_hits = sum(t_title.count(t) for t in q_tokens)
    sum_hits   = sum(t_sum.count(t)   for t in q_tokens)
    return title_hits, sum_hits

# domain synonyms for gentle expansion
SYNONYMS = {
    "cargo":    ["resupply","supplies","crs","dragon","station","iss"],
    "station":  ["iss","dragon","crs","resupply","cargo","international space station"],
    "resupply": ["crs","dragon","cargo","iss","station"],
    "crs":      ["resupply","dragon","cargo","iss","station"],
    "dragon":   ["crs","resupply","station","iss","cargo"],
    "iss":      ["station","dragon","crs","resupply","cargo"],
    "moon":     ["lunar","artemis"],
    "lunar":    ["moon","artemis"],
    "artemis":  ["moon","lunar"],
}

MUST_TERMS = {"cargo","resupply","dragon","crs","station","iss"}

def enhanced_keyword_score(query: str, title: str, summary: str = "") -> float:
    # base hits (title weighted 3x)
    t_hits, s_hits = keyword_hits(query, title, summary)
    score = 3.0 * t_hits + 1.0 * s_hits

    # exact-phrase bonus (e.g., "cargo to the station" appearing in title/summary)
    q_clean = " ".join(tokenize(query))
    if q_clean and q_clean in (title or "").lower():
        score += 2.0
    if q_clean and q_clean in (summary or "").lower():
        score += 0.7

    # synonym bonuses
    q_tokens = tokenize(query)
    t_low = (title or "").lower()
    s_low = (summary or "").lower()
    for tok in q_tokens:
        for syn in SYNONYMS.get(tok, []):
            if contains_word(t_low, syn):
                score += 1.0
            elif contains_word(s_low, syn):
                score += 0.3
    return score

def must_have_gate(query: str, title: str, summary: str) -> bool:
    """If query mentions any cargo/station terms, require at least one must-term in title or summary."""
    q_tokens = set(tokenize(query))
    if MUST_TERMS.intersection(q_tokens):
        for t in MUST_TERMS:
            if contains_word(title, t) or contains_word(summary, t):
                return True
        return False
    return True