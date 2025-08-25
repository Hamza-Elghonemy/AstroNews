import json, os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sentence_transformers import SentenceTransformer
from local_search import latest_jsonl_files,load_jsonl,textify,as_utc

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("EMBEDDING_MODEL")


def main():
    src = latest_jsonl_files()
    items = load_jsonl(src)
    print(f"Loaded {len(items)} items from {src.name}")

    model = SentenceTransformer(MODEL_NAME)
    texts = [textify(it) for it in items]

    embs = model.encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )
    embs = np.asarray(embs, dtype=np.float32)
    np.save(OUT_DIR / "embeddings.npy", embs)

    with (OUT_DIR / "meta.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            published_at = item.get("published_at")
            published_str = as_utc(published_at).isoformat() if published_at else None
            
            f.write(json.dumps({
                "title": item.get("title"),
                "url": item.get("url"),
                "published_at": published_str,
                "source": item.get("source")
            }, ensure_ascii=False) + "\n")

    (OUT_DIR / "model.txt").write_text(MODEL_NAME, encoding="utf-8")
    (OUT_DIR / "source_file.txt").write_text(src.name, encoding="utf-8")

    print("Saved:")
    print(" - data/index/embeddings.npy")
    print(" - data/index/meta.jsonl")
    print(" - data/index/model.txt")
    print(" - data/index/source_file.txt")
    
if __name__ == "__main__":
    main()