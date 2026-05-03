import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# model
MODEL_HUB = os.environ.get("MODEL_HUB", os.path.join(PROJECT_ROOT, "models"))
BGE_M3_PATH = os.environ.get("BGE_M3_PATH", os.path.join(MODEL_HUB, "bge-m3"))
BGE_RERANKER_PATH = os.environ.get("BGE_RERANKER_PATH", os.path.join(MODEL_HUB, "bge-reranker-v2-m3"))

# data and index
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "index")

# retrieval
RERANK_TOP_K = 5
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20