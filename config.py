import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = "/root/autodl-tmp"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model
MODEL_HUB = os.environ.get("MODEL_HUB", os.path.join(MODEL_ROOT, "models"))
BGE_M3_PATH = os.environ.get("BGE_M3_PATH", os.path.join(MODEL_HUB, "bge-m3"))
BGE_RERANKER_PATH = os.environ.get("BGE_RERANKER_PATH", os.path.join(MODEL_HUB, "bge-reranker-v2-m3"))

# data and index
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "index")

# retrieval
RERANK_TOP_K = 5
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20

# agents
AGENT_LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "qwen3.5-27b")
BASE_URL = os.environ.get("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.environ.get("API_KEY", "sk-44048d2fb666429eabb9f24864e382b3")