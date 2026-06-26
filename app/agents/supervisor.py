from langchain_core.messages import SystemMessage

from app.agents.constants import AGENTS_DESCRIPTION, AllowedAgents
from app.agents.states import ChatState
from app.llm import get_llm
from pydantic import BaseModel

SUPERVISOR_SYSTEM_PROMPT = f"""
Você é um agente responsavel por direcionar uma conversa para o agente correto.
Abaixo segue todos os agentes disponiveis (no formato "- <agent_key>") e suas respectivas descrições:

{''.join([f'- {agent}: {AGENTS_DESCRIPTION[agent]}' for agent in AllowedAgents])}

Seu objetivo é retornar o <agent_key> do agente que faz mais sentido para a conversa 
dado o histórico de mensagens e a descrição dos agentes.
"""


class Router(BaseModel):
    next: AllowedAgents
    message: str


async def supervisor_agent(state: ChatState):
    llm = get_llm().with_structured_output(Router)

    system_message = SystemMessage(SUPERVISOR_SYSTEM_PROMPT)
    messages = [system_message, *state['messages']]
    router = await llm.ainvoke(messages)

    assert isinstance(router, Router)
    return {'next': router.next}
