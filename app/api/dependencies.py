from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from app.agents.states import ChatState


def get_agent(
    request: Request,
) -> CompiledStateGraph[ChatState, None, ChatState, ChatState]:
    return request.app.state.agent


type Agent = CompiledStateGraph[ChatState, None, ChatState, ChatState]
