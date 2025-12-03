import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer

from .config import EMB_MODEL_NAME, CACHE_DIR

EMB_PATH = CACHE_DIR / "embeddings.npy"
META_PATH = CACHE_DIR / "meta.json"

def get_embedder():
    model = SentenceTransformer(EMB_MODEL_NAME, device="cpu")
    return model

def compute_and_cache_embeddings(chunks: List[Dict]):
    model = get_embedder()
    texts = [c["text"] for c in chunks]

    # batched encoding = big optimization on CPU
    embeddings = model.encode(
        texts,
        batch_size=32,  # experiment with 8 vs 16 vs 32
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMB_PATH, embeddings)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f)

def load_cached_embeddings():
    embeddings = np.load(EMB_PATH)
    with META_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, chunks
