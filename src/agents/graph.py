import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.router import route_query, route_decision
from agents.planner import plan
from agents.executor import TOOL_REGISTRY, _ALL_TOOLS, _ensure_tools
from agents.executor import execute_step, should_continue_executing
from agents.verifier import verify, after_verification
from agents.synthesizer import synthesize, simple_rag


def build_graph(enable_verifier: bool = True, enabled_tools: list[str] | None = None):
    """构建 AgenticRAG LangGraph 图

    Args:
        enable_verifier: 是否启用验证器（消融实验用）
        enabled_tools: 允许使用的工具列表（消融实验用）
    """
    _ensure_tools()
    TOOL_REGISTRY.clear()
    if enabled_tools is not None:
        for name in enabled_tools:
            if name in _ALL_TOOLS:
                TOOL_REGISTRY[name] = _ALL_TOOLS[name]
    else:
        TOOL_REGISTRY.update(_ALL_TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("router", route_query)
    graph.add_node("simple_rag", simple_rag)
    graph.add_node("planner", plan)
    graph.add_node("executor", execute_step)
    graph.add_node("synthesizer", synthesize)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "simple": "simple_rag",
            "multi_hop": "planner",
        },
    )

    graph.add_edge("simple_rag", END)
    graph.add_edge("planner", "executor")

    if enable_verifier:
        graph.add_node("verifier", verify)

        graph.add_conditional_edges(
            "executor",
            should_continue_executing,
            {
                "execute": "executor",
                "verify": "verifier",
            },
        )

        graph.add_conditional_edges(
            "verifier",
            after_verification,
            {
                "synthesize": "synthesizer",
                "replan": "planner",
            },
        )
    else:
        graph.add_conditional_edges(
            "executor",
            should_continue_executing,
            {
                "execute": "executor",
                "verify": "synthesizer",
            },
        )

    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_query(query: str, **kwargs) -> dict:
    """运行单个查询，返回完整 state"""
    app = build_graph(**kwargs)
    initial_state = {
        "query": query,
        "query_type": "multi_hop",
        "plan": [],
        "current_step": 0,
        "evidence": [],
        "tool_calls": [],
        "verification_result": "",
        "verification_feedback": "",
        "final_answer": "",
        "iteration_count": 0,
        "total_tool_calls": 0,
        "trace": [],
    }
    return app.invoke(initial_state)


_default_app = None


def get_default_app():
    global _default_app
    if _default_app is None:
        _default_app = build_graph()
    return _default_app
