import operator
from typing import TypedDict, Literal, Annotated


class AgentState(TypedDict):
    query: str # 用户查询
    query_type: Literal["simple", "multi_hop"] # 查询类型
    plan: list[str] # 子任务
    current_step: int # 当前步骤索引
    evidence: Annotated[list[dict], operator.add] # 累计证据（累加）
    tool_calls: Annotated[list[dict], operator.add] # 工具调用记录（累加）
    verification_result: str # 验证结果
    verification_feedback: str # 验证反馈
    final_answer: str # 最终答案
    iteration_count: int # 迭代次数
    total_tool_calls: int # 总工具调用次数
    trace: Annotated[list[dict], operator.add] # 跟踪记录（累加）
    