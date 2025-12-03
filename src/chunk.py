from typing import List, Dict


def chunk_document(doc: Dict, chunk_size: int, overlap: int) -> List[Dict]:
    text = doc["text"]
    chunks = []
    start = 0
    doc_id = doc["id"]

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "doc_id": doc_id,
            "start": start,
            "end": min(end, len(text)),
            "text": chunk_text
        })
        start += chunk_size - overlap

    return chunks

def chunk_corpus(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks
