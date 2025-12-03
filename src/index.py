import numpy as np
import faiss
from typing import Tuple, List, Dict

from sentence_transformers import SentenceTransformer
from .config import EMB_MODEL_NAME, TOP_K

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product; we normalized embeddings
    index.add(embeddings.astype(np.float32))
    return index

def get_query_embedder():
    
    return SentenceTransformer(EMB_MODEL_NAME, device="cpu")

def retrieve(
    question: str,
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
    embedder: SentenceTransformer,
    top_k: int = TOP_K
) -> List[Dict]:
    q_emb = embedder.encode(
        [question],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[int(idx)]
        chunk["score"] = float(score)
        results.append(chunk)
    return results
