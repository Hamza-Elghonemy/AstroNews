from pathlib import Path
import numpy as np
import faiss

EMB_PATH = Path("data/index/embeddings.npy")
IDX_PATH = Path("data/index/faiss.index")

def build_faiss_index():
    # Load embeddings
    embeddings = np.load(EMB_PATH)
    n,d = embeddings.shape

    print(f"Loaded embeddings from {EMB_PATH} of shape: {n} x {d}")

    # Build FAISS index
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, str(IDX_PATH))

    print(f"Saved FAISS index to {IDX_PATH} with {index.ntotal} vectors.")

if __name__ == "__main__":
    build_faiss_index()
