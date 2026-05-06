# 汽车行业年报多跳问答的Agentic RAG系统

# 1. 收集年报

主机厂：

​	奇瑞（HK）、比亚迪、吉利（HK）、长城、长安

零件厂：

​	宁德时代、福耀玻璃、德赛西威

```
rag-auto-report/
├── README.md                    # 背景、快速开始、评测摘要、局限
├── pyproject.toml / requirements.txt
├── .env.example                 # API Key、路径占位（勿提交真实密钥）
│
├── configs/                     # 单一配置入口
│   ├── default.yaml             # 路径、模型名、chunk、top_k、replan 上限等
│   └── train_grpo.yaml          # verl/GRPO 相关（可与 default 继承）
│
├── data/
│   ├── raw_pdfs/                # 原始年报 PDF（gitignore）
│   ├── processed/
│   │   └── chunks.jsonl         # 解析后的 chunk + metadata
│   └── eval/
│       ├── questions.jsonl      # 自拟评测题
│       └── labels.jsonl         # 可选：gold chunk / 短答
│
├── artifacts/                   # 生成物（通常 gitignore）
│   ├── faiss/                   # FAISS 索引
│   ├── bm25/                    # BM25 序列化
│   └── checkpoints/             # LoRA / GRPO 权重
│
├── src/
│   ├── ingest/                  # 解析与切片
│   │   ├── parse_pdf.py
│   │   └── chunking.py
│   ├── index/
│   │   ├── embeddings.py
│   │   ├── vector_store.py      # FAISS 读写
│   │   └── bm25_index.py
│   ├── retrieve/
│   │   ├── hybrid.py            # RRF / 融合
│   │   └── rerank.py            # 可选 CrossEncoder
│   ├── agent/
│   │   ├── graph.py             # LangGraph 编排
│   │   ├── nodes/               # planner / executor / verifier / synth
│   │   └── trace.py             # trace 记录（面试演示）
│   ├── llm/
│   │   ├── client.py            # 本地 vLLM / API 统一封装
│   │   └── prompts/             # 各节点 prompt 模板
│   ├── train/
│   │   ├── build_sft_data.py
│   │   ├── sft_lora.sh / sft_lora.py
│   │   └── grpo_verl.sh         # 或指到 verl 配置
│   ├── eval/
│   │   ├── run_baseline.py      # 单次 RAG
│   │   ├── run_agent.py
│   │   └── metrics.py           # 引用命中、F1 等
│   └── app/                     # 可选
│       └── demo.py              # Gradio/Streamlit
│
├── scripts/                     # 一键入口
│   ├── 01_ingest.sh
│   ├── 02_index.sh
│   ├── 03_run_eval.sh
│   └── 04_train_grpo.sh
│
├── notebooks/                   # 可选：探索性分析
│   └── eda_chunks.ipynb
│
└── tests/                       # 可选：小单测
    └── test_rrf.py
```

# 2. 解析PDF

遇到问题：

1. 匹配标题时需要考虑换行符等多种情况
2. 港股比A股的年报更难解析，需要单独处理
3. 表格难以处理

