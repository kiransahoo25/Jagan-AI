import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np

TEXT_EXTS = {
    ".py", ".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".json",
    ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp",
    ".h", ".hpp", ".sh", ".sql",
}
SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".idea", ".vscode",
}


def make_embedder(cfg):
    if cfg.embed_backend == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

        def embed(text: str):
            r = client.embeddings.create(model=cfg.embed_model, input=text)
            return r.data[0].embedding

        return embed

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.embed_model)
    return lambda t: model.encode(t, normalize_embeddings=True).tolist()


def chunk_file(path: Path, chunk_lines: int = 40, overlap: int = 10):
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return
    step = max(1, chunk_lines - overlap)
    for i in range(0, len(lines), step):
        chunk = "\n".join(lines[i : i + chunk_lines])
        if chunk.strip():
            yield i + 1, chunk


class VectorStore:
    def __init__(self, db_path: Path, embed_fn):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS chunks(
                id TEXT PRIMARY KEY,
                path TEXT,
                start INTEGER,
                text TEXT,
                vec BLOB)"""
        )
        self.db.commit()
        self.embed = embed_fn

    def add(self, path: Path, start: int, text: str):
        cid = hashlib.sha1(f"{path}:{start}".encode()).hexdigest()
        vec = np.asarray(self.embed(text), dtype=np.float32).tobytes()
        self.db.execute(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?)",
            (cid, str(path), start, text, vec),
        )

    def commit(self):
        self.db.commit()

    def search(self, query: str, k: int = 6):
        q = np.asarray(self.embed(query), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9

        scored = []
        for path, start, text, blob in self.db.execute(
            "SELECT path,start,text,vec FROM chunks"
        ):
            v = np.frombuffer(blob, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            scored.append((float(q @ v), path, start, text))

        scored.sort(key=lambda r: r[0], reverse=True)
        return scored[:k]


def ingest(root: Path, store: VectorStore, log=print):
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix not in TEXT_EXTS:
                continue
            try:
                if p.stat().st_size > 500_000:
                    continue
                for start, chunk in chunk_file(p):
                    store.add(p, start, chunk)
                    n += 1
            except Exception:
                continue
    store.commit()
    log(f"indexed {n} chunks")