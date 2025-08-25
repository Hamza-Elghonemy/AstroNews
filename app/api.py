from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import sys
import os
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

app = FastAPI(
    title="AstroNews Explorer API",
    description="Hybrid search API for astronomy news",
    version="1.0.0"
)

hybrid_search_func = None

@app.on_event("startup")
async def startup_event():
    """Initialize the search function on startup"""
    global hybrid_search_func
    try:
        from hybrid_search import hybrid_search
        hybrid_search_func = hybrid_search
        print("✅ Hybrid search initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize hybrid search: {e}")
        # Don't fail startup, just log the error

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "AstroNews Explorer API is running!",
        "status": "healthy",
        "search_available": hybrid_search_func is not None
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "search_function_loaded": hybrid_search_func is not None,
        "python_version": sys.version,
        "working_directory": str(Path.cwd())
    }

@app.get("/search")
def search(q: str = Query(..., description="Search query"),
           k: int = Query(5, description="Number of results to return")):
    """
    Run hybrid search on the news corpus.
    """
    if hybrid_search_func is None:
        raise HTTPException(
            status_code=503, 
            detail="Search service not available. Check if data files are present."
        )
    
    try:
        hits = hybrid_search_func(q, k=k)
        return {
            "query": q,
            "count": len(hits),
            "results": hits
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )
