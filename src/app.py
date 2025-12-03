import time
from pathlib import Path

from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from . import ingest, chunk as chunk_mod, embed, index, qa

def prepare_corpus():
    docs = ingest.load_documents(DATA_DIR)
    chunks = chunk_mod.chunk_corpus(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    embed.compute_and_cache_embeddings(chunks)

def interactive_qa():
    # load embeddings + chunks
    embeddings, chunks = embed.load_cached_embeddings()
    faiss_index = index.build_index(embeddings)
    embedder = index.get_query_embedder()
    qa_system = qa.QASystem()

    print("RAG-lite QA ready. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        t0 = time.time()
        retrieved = index.retrieve(question, faiss_index, chunks, embedder)
        contexts = [r["text"] for r in retrieved]
        mid = time.time()
        qa_result = qa_system.answer_from_contexts(question, contexts)
        t1 = time.time()

        print(f"\nAnswer: {qa_result['answer']}")
        print(f"(retrieval: {mid - t0:.3f}s, qa: {t1 - mid:.3f}s, total: {t1 - t0:.3f}s)")

def main():
    if not (Path("cache/embeddings.npy").exists()):
        print("No cache found. Preparing corpus and computing embeddings...")
        prepare_corpus()
    interactive_qa()

if __name__ == "__main__":
    main()
