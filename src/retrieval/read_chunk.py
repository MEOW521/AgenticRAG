import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INDEX_DIR

_chunk_store = None


def _load():
    global _chunk_store
    if _chunk_store is None:
        with open(os.path.join(INDEX_DIR, "chunk_store.pkl"), "rb") as f:
            _chunk_store = pickle.load(f)


def read_chunk(chunk_id: str) -> list[dict]:
    """按 chunk_id 读取完整 doc，返回 [{"chunk_id", "text", "title"}] 或空列表"""
    _load()
    doc = _chunk_store.get(chunk_id)
    if doc:
        return [{
            "chunk_id": doc["chunk_id"],
            "text": doc["text"],
            "title": doc.get("title", ""),
            "score": 1.0,
            "source": "read_chunk",
        }]
    return []
