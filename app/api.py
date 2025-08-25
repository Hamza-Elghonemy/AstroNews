from fastapi import FastAPI, Query
from typing import List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from hybrid_search import hybrid_search 

app = FastAPI(title="AstroNews Explorer API")

@app.get("/search")
def search(q: str = Query(..., description="Search query"),
           k: int = Query(5, description="Number of results to return")):
    """
    Run hybrid search on the news corpus.
    """
    hits = hybrid_search(q, k=k)
    return {
        "query": q,
        "count": len(hits),
        "results": hits
    }
