from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from sse_starlette.sse import EventSourceResponse

from app.api.deps import Agent, get_agent
from app.api.schemas.chat import ChatRequest
from app.core.services import chat as chat_service

router = APIRouter(prefix='/threads', tags=['chat'])


@router.post('/{thread_id}/chat')
async def chat(
    thread_id: str,
    req: ChatRequest,
    agent: Agent = Depends(get_agent),
) -> EventSourceResponse:
    config: RunnableConfig = {'configurable': {'thread_id': thread_id}}

    # Turn-based: todo turno é uma mensagem nova rodando START→END. A posição no
    # fluxo multi-turno vive no `phase` do state (lida pelo supervisor), então não
    # há mais interrupt para retomar via Command(resume=...).
    graph_input = {'messages': [HumanMessage(req.message)], 'next': ''}

    return EventSourceResponse(chat_service.event_stream(agent, graph_input, config))
