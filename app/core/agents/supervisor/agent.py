from langchain_core.messages import SystemMessage

from app.core.agents.constants import PHASE_OWNER, Phase
from app.core.agents.states import ChatState
from app.core.agents.supervisor.prompt import SUPERVISOR_SYSTEM_PROMPT
from app.core.agents.supervisor.schemas import Router
from app.infra.llm import get_llm


async def build_supervisor_agent(state: ChatState) -> ChatState:
    # Override determinístico: se há uma fase multi-turno ativa, o dono dela já é
    # conhecido (PHASE_OWNER) — roteia direto, sem gastar uma chamada de LLM e sem
    # risco de vazar para outro agente no meio do fluxo.
    phase = state.get('phase')
    if phase:
        return {'messages': [], 'next': PHASE_OWNER[Phase(phase)].value}

    llm = get_llm().with_structured_output(Router)

    system_message = SystemMessage(SUPERVISOR_SYSTEM_PROMPT)
    messages = [system_message, *state['messages']]
    router = await llm.ainvoke(messages)

    assert isinstance(router, Router)
    return {'messages': [], 'next': router.next.value}
