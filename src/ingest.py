from pathlib import Path
from typing import List, Dict

def load_documents(data_dir: Path) -> List[Dict]:
    docs = []
    for path in data_dir.glob("**/*"):
        if path.suffix.lower() not in [".txt", ".md"]:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append({"id": path.stem, "path": str(path), "text": text})
    return docs
