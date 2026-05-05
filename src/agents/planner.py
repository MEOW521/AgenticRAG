import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import AgentState
from agents.prompts import get_profiles
from llms.clients import agent_chat_json

TOOL_DESCRIPTIONS = {
    "keyword_search": "BM25 keyword search, good for exact names/entities",
    "semantic_search": "Dense retrieval + reranking, good for semantic similarity",
    "read_chunk": "Read a specific document by chunk_id (use when you have a specific chunk_id from previous steps)",
}


def plan(state: AgentState):
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)
    profile = get_profiles()

    feedback_section = ""
    if iteration_count > 0 and state.get("verification_feedback"):
        evidence_summary = "\n".join(
            f"- Step {e['step_id']} [{e.get('tool', '?')}]: \"{e['sub_query']}\" -> {len(e.get('results', []))} results"
            for e in state.get("evidence", [])
        )
        feedback_section = profile["replan_feedback"].format(
            feedback=state["verification_feedback"],
            evidence_summary=evidence_summary or "No evidence found",
        )

    from agents.executor import TOOL_REGISTRY, _ensure_tools

    _ensure_tools()
    tools_section = "\n".join(
        f"- {name}: {TOOL_DESCRIPTIONS.get(name, 'search tool')}" for name in TOOL_REGISTRY
    )
    if not tools_section:
        tools_section = "- semantic_search: Dense retrieval + reranking"

    prompt = profile["planner"].format(
        query=query, feedback_section=feedback_section, tools_section=tools_section
    )
    result = agent_chat_json(prompt)

    if not result or not isinstance(result, list):
        result = [{"id": 1, "sub_query": query, "tool": "semantic_search", "depends_on": []}]

    for step in result:
        step["status"] = "pending"

    return {
        "plan": result,
        "current_step": 0,
        "iteration_count": iteration_count + 1,
        "trace": [
            {"node": "planner", "iteration_count": iteration_count + 1, "plan": result}
        ],
    }
