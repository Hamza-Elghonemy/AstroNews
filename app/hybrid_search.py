import os, json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from datetime import datetime, timezone
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from text_utils import enhanced_keyword_score, must_have_gate 
from semantic_search import load_meta
from local_search import recency_boost
from score_plot import save_debug_plots, plot_breakdown
from dotenv import load_dotenv



load_dotenv()

IDX_PATH    = Path("data/index/faiss.index")
META_PATH   = Path("data/index/meta.jsonl")
SOURCE_FILE = Path("data/index/source_file.txt")
RAW_DIR     = Path("data/raw")
MODEL_NAME  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Blending weights
SEM_W, KW_W, REC_W = 0.45, 0.35, 0.20  # semantic / keyword / recency


def load_raw_items() -> list[dict]:
    """Load the original raw JSONL (with summaries), aligned with embeddings order."""
    src_name = SOURCE_FILE.read_text(encoding="utf-8").strip()
    src_path = RAW_DIR / src_name
    items: list[dict] = []
    with src_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def semantic_candidates(query: str, k: int = 20):
    """Return (scores, indices) from FAISS for the top-k semantic neighbors."""
    index = faiss.read_index(str(IDX_PATH))
    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q, k)
    return D[0], I[0]


def hybrid_search(query: str, k: int = 8):
    """
    Hybrid = normalized semantic + normalized keyword + recency, with a soft must-have gate.
    """
    meta      = load_meta()
    raw_items = load_raw_items()
    now       = datetime.now(timezone.utc)

    cand_k = min(50, len(meta))
    sem_scores, sem_idxs = semantic_candidates(query, k=cand_k)

    candidates = []
    for s, idx in zip(sem_scores, sem_idxs):
        if idx == -1 or idx >= len(meta):
            continue
        m   = meta[idx]
        raw = raw_items[idx] if idx < len(raw_items) else {}

        title   = m.get("title") or ""
        summary = raw.get("summary") or ""
        pub_at  = m.get("published_at") or m.get("published")

        kw_raw = enhanced_keyword_score(query, title, summary)
        rec    = recency_boost(pub_at, now) 

        candidates.append({
            "idx": int(idx),
            "title": title,
            "summary": summary,
            "url": m.get("url"),
            "published_at": pub_at,
            "source": m.get("source"),
            "semantic_raw": float(s),
            "keyword_raw":  float(kw_raw),
            "recency":      float(rec),
        })

    if not candidates:
        return []

    # normalize semantic (min-max) to [0,1]
    sem_vals = [c["semantic_raw"] for c in candidates]
    smin, smax = float(min(sem_vals)), float(max(sem_vals))
    sden = max(1e-9, smax - smin)
    for c in candidates:
        c["semantic_norm"] = (c["semantic_raw"] - smin) / sden

    # normalize keyword by max to [0,1]
    kw_vals = [c["keyword_raw"] for c in candidates]
    kmax = max(1e-9, float(max(kw_vals)))
    for c in candidates:
        c["keyword_norm"] = c["keyword_raw"] / kmax

    # final score + soft must-have
    for c in candidates:
        score = SEM_W * c["semantic_norm"] + KW_W * c["keyword_norm"] + REC_W * c["recency"]
        if not must_have_gate(query, c["title"], c["summary"]):
            score *= 0.6  # soft penalty if query expects cargo/station but doc lacks it
        c["score_final"] = score

    candidates.sort(key=lambda x: x["score_final"], reverse=True)

    results = []
    for c in candidates[:k]:
        results.append({
            "title": c["title"],
            "url": c["url"],
            "published_at": c["published_at"],
            "source": c["source"],
            "score_final":    float(c["score_final"]),
            "score_semantic": float(c["semantic_norm"]),
            "score_keyword":  float(c["keyword_norm"]),
            "score_recency":  float(c["recency"]),
            "semantic_raw":   float(c["semantic_raw"]),
            "keyword_raw":    float(c["keyword_raw"]),
        })
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Hybrid search (semantic + keyword + recency)")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--debug", type=int, default=0, help="Number of debug plots to generate (0 = none)")
    ap.add_argument("--out-dir", "-o", type=str, default="debug_plots", help="Output directory for debug plots")
    args = ap.parse_args()

    hits = hybrid_search(args.query, k=args.k)
    
    print(f"\nTop {len(hits)} hybrid results for: {args.query!r}\n")
    for i, h in enumerate(hits, 1):
        date = (h.get("published_at") or "")[:10]
        print(f"[{i}] {date}  "
              f"(final={h['score_final']:.3f} | sem={h['score_semantic']:.3f} | kw={h['score_keyword']:.3f} | rec={h['score_recency']:.3f})")
        print(f"     {h['title']}")
        print(f"     {h['url']}")
    
    if args.debug > 0:
        # reconstruct minimal candidate dicts from 'hits'
        candidates_sorted = [{
            "title": h["title"],
            "semantic_norm": h["score_semantic"],
            "keyword_norm":  h["score_keyword"],
            "recency":       h["score_recency"],
        } for h in hits]
        save_debug_plots(
            candidates_sorted,
            top_n=args.debug,
            out_dir=Path(args.out_dir),
            sem_w=SEM_W, kw_w=KW_W, rec_w=REC_W
        )
        print(f"\nSaved {min(args.debug, len(hits))} breakdown chart(s) to {args.out_dir}")