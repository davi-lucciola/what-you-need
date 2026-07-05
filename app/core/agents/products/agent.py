from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from app.core.agents.constants import Phase
from app.core.agents.products.constants import PRODUCT_AGENT_RECURSION_LIMIT
from app.core.agents.products.prompt import PRODUCT_AGENT_SYSTEM_PROMPT
from app.core.agents.products.tools import build_product_tools
from app.core.agents.states import ChatState
from app.infra.llm import get_llm

# Nome da tool terminal: sua presença no turno sinaliza que o fluxo terminou.
PRESENT_TOOL = 'present_recommendations'
# Fallback improvável: agente não produziu pergunta nem apresentação neste turno.
_FALLBACK_QUESTION = 'Pode me contar um pouco mais sobre o que você procura?'


def _last_ai_text(messages: list[AnyMessage]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.text.strip():
            return message.text

    return None


def _present_output(messages: list[AnyMessage]) -> str | None:
    """Conteúdo da tool terminal `present_recommendations`, se chamada neste turno."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == PRESENT_TOOL:
            content = message.content
            return content if isinstance(content, str) else str(content)

    return None


def _turn_delta(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Mensagens novas deste turno: tudo após o último HumanMessage.

    Robusto a qualquer prefixo que o agente insira e, crucialmente, isola a detecção
    da tool terminal ao turno atual (evita re-apresentar por causa de um
    `present_recommendations` de um fluxo anterior ainda no histórico).
    """
    indices = range(len(messages))
    last_human = next(
        (i for i in reversed(indices) if isinstance(messages[i], HumanMessage)), -1
    )
    return messages[last_human + 1 :]


async def product_agent(state: ChatState) -> dict[str, Any]:
    messages = list(state['messages'])

    agent = create_agent(
        model=get_llm(),
        tools=build_product_tools(),
        system_prompt=PRODUCT_AGENT_SYSTEM_PROMPT,
    )
    result = await agent.ainvoke(
        cast(Any, {'messages': messages}),
        config={'recursion_limit': PRODUCT_AGENT_RECURSION_LIMIT},
    )
    delta = _turn_delta(cast(list[AnyMessage], result['messages']))

    presentation = _present_output(delta)
    if presentation is not None:
        return {'messages': [AIMessage(presentation)], 'phase': ''}

    question = _last_ai_text(delta) or _FALLBACK_QUESTION
    return {'messages': [AIMessage(question)], 'phase': Phase.PRODUCTS_ACTIVE.value}
