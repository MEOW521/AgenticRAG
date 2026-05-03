import json
import os
import pickle
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# faiss
import faiss
from retrieval.embedding import encode

# bm25
from rank_bm25 import BM25Okapi
from retrieval.keyword_search import tokenize



def index_all(corpus_path: str = None, index_dir: str = None):
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"[Indexing] Building indexes for {len(corpus)} chunks...")

    texts = [doc["text"] for doc in corpus]
    chunk_ids = [doc["chunk_id"] for doc in corpus]
    chunk_store = {doc["chunk_id"]: doc for doc in corpus}

    # 1. FAISS IndexFlatIP
    print("[Indexing] Encoding with BGE-M3...")
    embeddings = encode(texts, batch_size=64)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    print(f"[Indexing] FAISS index: {index.ntotal} vectors, dim={dim}")

    # 2. BM25
    print("[Indexing] Building BM25...")
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    # 3. chunk_ids（与 FAISS 对齐）
    with open(os.path.join(index_dir, "chunk_ids.json"), "w") as f:
        json.dump(chunk_ids, f)

    # 4. chunk_store
    with open(os.path.join(index_dir, "chunk_store.pkl"), "wb") as f:
        pickle.dump(chunk_store, f)   


if __name__ == "__main__":
    # index_all(corpus_path="data/corpus/corpus_all.json", index_dir="data/index")
    index_all(corpus_path="data/datasets/corpus.json", index_dir="data/datasets/index")