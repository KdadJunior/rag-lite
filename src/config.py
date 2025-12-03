from pathlib import Path

DATA_DIR = Path("data/raw_docs")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "distilbert-base-uncased-distilled-squad"

CHUNK_SIZE = 400      # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks
TOP_K = 5             # retrieved chunks per question

