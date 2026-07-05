from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pytest_mock import MockerFixture

from app.core.agents.constants import Phase
from app.core.agents.products import agent as products_agent
from app.core.agents.products import build_product_search_node
from app.core.agents.products.agent import PRESENT_TOOL, product_agent


def _fake_agent(mocker: MockerFixture, new_messages: list[Any]) -> Any:
    """Agente falso cujo `ainvoke` devolve os `messages` de entrada + os novos."""
    agent = mocker.Mock()

    async def ainvoke(payload: dict[str, Any], *_: Any, **__: Any):
        return {'messages': [*payload['messages'], *new_messages]}

    agent.ainvoke = mocker.AsyncMock(side_effect=ainvoke)
    return agent


def _patch_agent(mocker: MockerFixture, new_messages: list[Any]) -> Any:
    mocker.patch.object(products_agent, 'get_llm', return_value=mocker.Mock())
    agent = _fake_agent(mocker, new_messages)
    create_agent = mocker.patch.object(
        products_agent, 'create_agent', return_value=agent
    )
    return create_agent


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_build_product_search_node_returns_the_agent_node() -> None:
    # O nó de produtos agora é uma única função (ReAct agent), não um subgrafo.
    assert build_product_search_node() is product_agent


# --------------------------------------------------------------------------- #
# Turno de pergunta: agente responde texto → mantém a fase ativa.
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_question_turn_keeps_phase_active(mocker: MockerFixture) -> None:
    _patch_agent(mocker, [AIMessage('Que produto você procura e para quê?')])

    state: Any = {'messages': [HumanMessage('quero comprar algo')]}
    result = await product_agent(state)

    assert result['phase'] == Phase.PRODUCTS_ACTIVE.value
    assert result['messages'][0].content == 'Que produto você procura e para quê?'


# --------------------------------------------------------------------------- #
# Turno final: tool `present_recommendations` chamada → limpa a fase e apresenta.
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_present_turn_clears_phase_and_emits_presentation(
    mocker: MockerFixture,
) -> None:
    new_messages = [
        AIMessage('', tool_calls=[{'name': PRESENT_TOOL, 'args': {}, 'id': 't1'}]),
        ToolMessage(
            content='LISTA FINAL DE PRODUTOS', name=PRESENT_TOOL, tool_call_id='t1'
        ),
        AIMessage('Prontinho, aqui estão!'),  # tagarelice final é ignorada
    ]
    _patch_agent(mocker, new_messages)

    state: Any = {'messages': [HumanMessage('até 2000 reais')]}
    result = await product_agent(state)

    assert result['phase'] == ''
    assert result['messages'][0].content == 'LISTA FINAL DE PRODUTOS'


# --------------------------------------------------------------------------- #
# Fallback improvável: agente não produziu pergunta nem apresentação.
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_empty_turn_falls_back_to_question(mocker: MockerFixture) -> None:
    _patch_agent(mocker, [AIMessage('')])

    state: Any = {'messages': [HumanMessage('oi')]}
    result = await product_agent(state)

    assert result['phase'] == Phase.PRODUCTS_ACTIVE.value
    assert result['messages'][0].content  # mensagem de fallback não-vazia
