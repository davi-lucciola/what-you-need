from langchain_core.messages import SystemMessage

from app.agents.states import ChatState
from app.agents.supervisor.prompt import SUPERVISOR_SYSTEM_PROMPT
from app.agents.supervisor.schemas import Router
from app.llm import get_llm


async def build_supervisor_agent(state: ChatState) -> ChatState:
    llm = get_llm().with_structured_output(Router)

    system_message = SystemMessage(SUPERVISOR_SYSTEM_PROMPT)
    messages = [system_message, *state['messages']]
    router = await llm.ainvoke(messages)

    assert isinstance(router, Router)
    return {'messages': [], 'next': router.next.value}
