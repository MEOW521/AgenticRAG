"""合成器：基于证据生成最终答案"""
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import AgentState
from agents.prompts import get_profiles
from llms.clients import agent_chat
from retrieval.semantic_search import semantic_search


def _extract_short_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    for prefix in [
        "Answer:",
        "answer:",
        "The answer is",
        "the answer is",
        "Based on the evidence,",
        "Based on the context,",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    first_line = text.split("\n")[0].strip()
    if len(first_line) > 150:
        for sep in [". ", "。", "; "]:
            if sep in first_line:
                first_line = first_line[: first_line.index(sep)].strip()
                break
    return first_line.rstrip(".")


def synthesize(state: AgentState) -> AgentState:
    query = state["query"]
    evidence = state.get("evidence", [])

    evidence_text = ""
    for e in evidence:
        for r in e.get("results", [])[:3]:
            evidence_text += f"[{r.get('chunk_id', '?')}] {r.get('text', '')[:500]}\n\n"

    profile = get_profiles()
    prompt = profile["synthesizer"].format(
        query=query, evidence_text=evidence_text or "No evidence available."
    )
    answer = agent_chat(prompt)
    answer = _extract_short_answer(answer or "")

    return {
        "final_answer": answer,
        "trace": [{"node": "synthesizer", "answer_length": len(answer)}],
    }


def simple_rag(state: AgentState) -> AgentState:
    query = state["query"]
    results = semantic_search(query)

    profile = get_profiles()
    context = "\n\n".join(r["text"][:500] for r in results[:3])
    prompt = profile["simple_rag"].format(context=context, query=query)
    answer = agent_chat(prompt)
    answer = _extract_short_answer(answer or "")

    evidence = [
        {
            "step_id": 1,
            "sub_query": query,
            "tool": "semantic_search",
            "results": results[:5],
        }
    ]

    return {
        "final_answer": answer.strip(),
        "evidence": evidence,
        "total_tool_calls": 1,
        "trace": [{"node": "simple_rag", "num_results": len(results)}],
    }
