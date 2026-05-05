from config import AGENT_LLM_MODEL

### Planner Prompts ###
_PLANNER_PROMPT_SMALL = """你是一个多跳问答系统的查询分解规划器。

将复杂问题拆解为可通过知识库检索逐一回答的子查询序列。

问题：{query}

{feedback_section}

每个子查询可用的工具：
{tools_section}

以 JSON 数组格式回复：
[
  {{"id": 1, "sub_query": "...", "tool": "<工具名>", "depends_on": []}},
  {{"id": 2, "sub_query": "...", "tool": "<工具名>", "depends_on": [1]}},
  ...
]

重要规则：
- 计划尽量精简。每步只检索一个具体实体或事实，只添加真正必要的步骤。
- 默认使用 semantic_search。仅在搜索非常具体的专有名词或代码时使用 keyword_search。
- 对召回率要求高的关键步骤，可以指定多个工具（如 "tool": ["semantic_search", "keyword_search"]）来融合多路检索结果。
- 不要创建用不同工具搜索相同内容的冗余步骤。
- 不要添加验证或确认步骤——只搜索所需的事实。"""

_PLANNER_PROMPT_LARGE = """你是一个多跳问答系统的查询分解规划器。

将复杂问题拆解为可通过知识库检索逐一回答的子查询序列。

问题：{query}

{feedback_section}

每个子查询可用的工具：
{tools_section}

以 JSON 数组格式回复：
[
  {{"id": 1, "sub_query": "...", "tool": "<工具名>", "depends_on": []}},
  {{"id": 2, "sub_query": "...", "tool": "<工具名>", "depends_on": [1]}},
  ...
]

重要规则：
- 计划尽量精简。每步只检索一个具体实体或事实，只添加真正必要的步骤。
- 为每个子查询选择最佳工具：精确名称/代码用 keyword_search，概念性或描述性查询用 semantic_search。
- 对召回率要求高的关键步骤，可以指定多个工具（如 "tool": ["semantic_search", "keyword_search"]）来融合多路检索结果。
- 不要创建用不同工具搜索相同内容的冗余步骤。
- 不要添加验证或确认步骤——只搜索所需的事实。"""

### Verifier Prompts ###
_VERIFIER_PROMPT_SMALL = """你是一个多跳问答系统的证据充分性验证器。

原始问题：{query}

已收集的证据：
{evidence_text}

评估已收集的证据是否足以回答原始问题。

重要：请宽松判定。如果证据包含问题中提到的关键实体的相关信息，就判定为"sufficient"。仅当关键实体在所有证据中完全缺失时才判定为"insufficient"。

以 JSON 格式回复：
{{
  "verdict": "sufficient" 或 "insufficient",
  "reasoning": "简要评估",
  "feedback": "如果 insufficient，具体缺少什么实体/事实？"
}}"""

_VERIFIER_PROMPT_LARGE = """你是一个多跳问答系统的证据充分性验证器。

原始问题：{query}

已收集的证据：
{evidence_text}

评估已收集的证据是否足以回答原始问题。

检查以下几点：
1. 证据是否覆盖了问题中提到的所有关键实体。
2. 回答所需的具体事实（如国籍、日期、属性）是否存在。
3. 对于比较类问题，是否每个被比较的实体都有对应证据。

以 JSON 格式回复：
{{
  "verdict": "sufficient" 或 "insufficient",
  "reasoning": "简要评估",
  "feedback": "如果 insufficient，具体缺少什么实体/事实？"
}}"""

### Synthesizer Prompts ###
_SYNTHESIZER_PROMPT = """你是一个问答基准测试的答案合成器。根据收集到的证据回答问题。

问题：{query}

证据：
{evidence_text}

关键规则：
- 只输出答案本身，不要输出其他任何内容
- 答案应该是简短的实体、名称、数字、是/否或简短短语
- 不要解释推理过程
- 不要添加限定词、注意事项或"根据证据"等短语
- 不要重复问题
- 好的答案示例："巴黎"、"是"、"42"、"爱因斯坦"、"蓝色的那个"

答案："""

### SimpleRAG Prompts ###
_SIMPLE_RAG_PROMPT = """根据以下上下文回答问题。只输出简短的短语或实体作为答案，不要解释。

上下文：
{context}

问题：{query}

答案："""

### Router Prompts ###
_ROUTER_PROMPT = """你是一个查询复杂度分类器。分析问题并判断它需要：
- "simple"：单次检索即可回答（单一事实查找）
- "multi_hop"：需要多次检索、比较或跨文档推理

问题：{query}

以 JSON 格式回复：{{"query_type": "simple" 或 "multi_hop", "reasoning": "简要说明"}}"""

### Replan Prompts ###
_REPLAN_FEEDBACK = """上一轮计划不充分。验证器反馈：
{feedback}

已尝试的检索（工具 + 查询）：
{evidence_summary}

请制定改进的计划来弥补不足。重要：
- 不要重复上面列出的相同工具+查询组合。
- 尝试不同的工具（如从 semantic_search 切换到 keyword_search 或 graph_search）或用不同的关键词重新表述查询。
- 聚焦于反馈中提到的具体缺失实体/事实。"""

PROMPT_PROFILES = {
    "small": {
        "planner": _PLANNER_PROMPT_SMALL,
        "verifier": _VERIFIER_PROMPT_SMALL,
        "synthesizer": _SYNTHESIZER_PROMPT,
        "simple_rag": _SIMPLE_RAG_PROMPT,
        "router": _ROUTER_PROMPT,
        "replan_feedback": _REPLAN_FEEDBACK,
        "max_iterations": 3,
        "max_retrieval_calls": 15,
    },
    "large": {
        "planner": _PLANNER_PROMPT_LARGE,
        "verifier": _VERIFIER_PROMPT_LARGE,
        "synthesizer": _SYNTHESIZER_PROMPT,
        "simple_rag": _SIMPLE_RAG_PROMPT,
        "router": _ROUTER_PROMPT,
        "replan_feedback": _REPLAN_FEEDBACK,
        "max_iterations": 5,
        "max_retrieval_calls": 15,
    },
}

def get_profiles(model_name: str=None):
  if model_name is None:
    model_name = AGENT_LLM_MODEL
  if model_name == "Qwen3-8B":
    return PROMPT_PROFILES["small"]
  else:
    return PROMPT_PROFILES["large"]
