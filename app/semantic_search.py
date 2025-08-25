import os, json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

IDX_PATH = Path("data/index/faiss.index")
META_PATH = Path("data/index/meta.jsonl")
MODEL_NAME = os.getenv("EMBEDDING_MODEL")

def load_meta():
    rows = []
    with META_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def search(query:str, k:int=5):
    meta = load_meta()
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(IDX_PATH))
    
    
    query_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(query_embedding, k) # Search the index

    results = []    
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        m = meta[idx]
        results.append({
            "score": float(score),
            "title": m["title"],
            "url": m["url"],
            "published_at": m["published_at"],
            "source": m["source"]
        })
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Semantic search with FAISS")
    ap.add_argument("--query", type=str, required=True, help="Search query")
    ap.add_argument("--k", type=int, default=5, help="Number of results to return")
    args = ap.parse_args()

    results = search(args.query, args.k)
    print(f"Top {len(results)} results for query: {args.query!r}\n")
    for i,h in enumerate(results, 1):
        date = (h.get("published_at") or "")[:10]
        print(f"[{i}] {date} — {h.get('title')}")
        print(f"     {h.get('url')} (score: {h.get('score'):.4f})")
        print()