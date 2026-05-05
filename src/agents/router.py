import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.state import AgentState
from agents.prompts import get_profiles
from llms.clients import agent_chat_json


def route_query(state: AgentState):
    query = state["query"]
    profile = get_profiles()
    result = agent_chat_json(profile["router"].format(query=query))

    query_type = "multi_hop"
    if result and result.get("query_type") == "simple":
        query_type = "simple"

    return {
        "query_type": query_type,
        "trace": [
            {
                "node": "router",
                "query_type": query_type,
                "detail": result,
            }
        ],
    }


def route_decision(state: AgentState):
    return state.get("query_type", "multi_hop")
