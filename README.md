# RAG-Lite

A lightweight Retrieval-Augmented Generation (RAG) system built with Python, designed for question-answering over local documents. This implementation provides a simple yet effective approach to RAG using sentence transformers for embeddings and FAISS for similarity search.

## Overview

RAG-Lite enables you to:
- Ingest and process local text documents
- Generate embeddings and build a searchable index
- Answer questions based on your document corpus using an interactive CLI

The system uses a two-stage approach: **retrieval** (finding relevant document chunks) and **generation** (extracting answers from retrieved context).

## Features

- 📄 **Document Processing**: Loads and chunks text documents with configurable overlap
- 🔍 **Semantic Search**: Uses sentence transformers for high-quality embeddings
- ⚡ **Fast Retrieval**: FAISS-based similarity search for efficient document retrieval
- 🤖 **Question Answering**: DistilBERT-based QA model for accurate answer extraction
- 💾 **Caching**: Embeddings are cached to avoid recomputation
- 🎯 **Interactive CLI**: Simple command-line interface for asking questions

## Architecture

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│   Chunking  │────▶│  Embeddings  │
└─────────────┘     └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  FAISS Index │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Retrieval   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  QA System   │
                    └──────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/KdadJunior/rag-lite.git
cd rag-lite
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: Models will be automatically downloaded from Hugging Face on first run.

## Usage

### 1. Prepare Your Documents

Place your text documents (`.txt` or `.md` files) in the `data/raw_docs/` directory:

```
data/
└── raw_docs/
    ├── notes1.txt
    ├── notes2.txt
    └── ...
```

### 2. Run the Application

```bash
python -m src.app
```

On first run, the system will:
- Load all documents from `data/raw_docs/`
- Chunk the documents
- Compute embeddings (this may take a few minutes)
- Cache the embeddings for future use

### 3. Ask Questions

Once ready, you'll see:
```
RAG-lite QA ready. Type 'exit' to quit.

Question: 
```

Type your question and press Enter to get an answer. The system will show:
- The answer extracted from your documents
- Timing information (retrieval time, QA time, total time)

Example:
```
Question: What is machine learning?

Answer: Machine learning is a subset of artificial intelligence...
(retrieval: 0.023s, qa: 0.156s, total: 0.179s)
```

To exit, type `exit` or `quit`.

## Configuration

Edit `src/config.py` to customize the system:

```python
DATA_DIR = Path("data/raw_docs")          # Directory containing documents
CHUNK_SIZE = 400                          # Characters per chunk
CHUNK_OVERLAP = 100                       # Overlap between chunks
TOP_K = 5                                 # Number of chunks to retrieve
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
QA_MODEL_NAME = "distilbert-base-uncased-distilled-squad"  # QA model
```

### Model Options

- **Embedding Models**: Any sentence transformer model from Hugging Face
  - Recommended: `all-MiniLM-L6-v2` (fast, good quality)
  - Larger: `all-mpnet-base-v2` (slower, better quality)

- **QA Models**: Any Hugging Face question-answering model
  - Default: `distilbert-base-uncased-distilled-squad` (lightweight)
  - Alternative: `bert-large-uncased-whole-word-masking-finetuned-squad`

## Project Structure

```
rag-lite/
├── data/
│   └── raw_docs/          # Place your documents here
├── cache/                 # Cached embeddings (auto-generated)
├── src/
│   ├── app.py            # Main application entry point
│   ├── chunk.py          # Document chunking logic
│   ├── config.py         # Configuration settings
│   ├── embed.py          # Embedding computation and caching
│   ├── index.py          # FAISS index and retrieval
│   ├── ingest.py         # Document loading
│   └── qa.py             # Question-answering system
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## How It Works

1. **Document Ingestion**: Text files are loaded from the data directory
2. **Chunking**: Documents are split into overlapping chunks for better context preservation
3. **Embedding**: Each chunk is converted to a vector using a sentence transformer
4. **Indexing**: Embeddings are stored in a FAISS index for fast similarity search
5. **Retrieval**: When a question is asked, it's embedded and used to find the most relevant chunks
6. **Answer Generation**: The QA model extracts an answer from the retrieved context

## Performance Tips

- **First Run**: Initial embedding computation may take several minutes depending on corpus size
- **Subsequent Runs**: Embeddings are cached, so startup is much faster
- **Large Documents**: Consider adjusting `CHUNK_SIZE` and `CHUNK_OVERLAP` for optimal results
- **Retrieval Quality**: Increase `TOP_K` if answers seem incomplete (at the cost of speed)

## Limitations

- Currently supports only `.txt` and `.md` files
- Answer quality depends on the relevance of retrieved chunks
- Best suited for factual questions over well-structured documents
- Single-turn Q&A (no conversation history)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Hugging Face Transformers](https://huggingface.co/transformers/) for QA models

## Future Improvements

- [ ] Support for PDF and other document formats
- [ ] Web UI interface
- [ ] Multi-turn conversation support
- [ ] Streaming responses
- [ ] Batch question processing
- [ ] Export/import of indices

---

Made with ❤️ for simple and effective RAG applications.
