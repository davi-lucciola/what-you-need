from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pytest_mock import MockerFixture

from app.core.agents.constants import Nodes
from app.core.services.chat import (
    _final_message,  # pyright: ignore[reportPrivateUsage]
    _is_streamable_token,  # pyright: ignore[reportPrivateUsage]
    event_stream,
)

CONFIG: Any = {'configurable': {'thread_id': 't1'}}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def make_agent(
    mocker: MockerFixture,
    *,
    stream_items: list[Any] | None = None,
    state_messages: list[Any] | None = None,
) -> Any:
    """Monta um agente compilado mockado.

    - `agent.astream(...)` é iterado com `async for` (stream_mode='messages'), então
      cada item é uma tupla `(chunk, metadata)`.
    - `agent.aget_state(...)` é awaited e retorna um snapshot com `.values`.
    """
    snapshot = SimpleNamespace(values={'messages': state_messages or []})
    agent = mocker.Mock()
    agent.astream = mocker.Mock(return_value=_aiter(stream_items or []))
    agent.aget_state = mocker.AsyncMock(return_value=snapshot)
    return agent


def messages_chunk(text: str, node: str = Nodes.GUIDE) -> Any:
    """Item de stream no modo `'messages'`: `(chunk, metadata)`."""
    return AIMessageChunk(content=text), {'langgraph_node': node}


# --------------------------------------------------------------------------- #
# _final_message (async)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_final_message_picks_last_ai_message_with_content(
    mocker: MockerFixture,
) -> None:
    agent = make_agent(
        mocker,
        state_messages=[
            HumanMessage('quero um celular'),
            AIMessage('primeira'),
            AIMessage(''),  # sem conteúdo -> ignorada
            AIMessage('última'),
            HumanMessage('obrigado'),  # não é AIMessage -> ignorada
        ],
    )

    result = await _final_message(agent, CONFIG)

    assert result == 'última'
    agent.aget_state.assert_awaited_once_with(CONFIG)


@pytest.mark.anyio
async def test_final_message_none_when_no_ai_content(mocker: MockerFixture) -> None:
    agent = make_agent(
        mocker,
        state_messages=[HumanMessage('oi'), AIMessage('')],
    )

    assert await _final_message(agent, CONFIG) is None


@pytest.mark.anyio
async def test_final_message_none_when_no_messages(mocker: MockerFixture) -> None:
    agent = make_agent(mocker, state_messages=[])

    assert await _final_message(agent, CONFIG) is None


# --------------------------------------------------------------------------- #
# event_stream (async generator — núcleo)
# --------------------------------------------------------------------------- #
async def _collect(agent: Any) -> list[dict[str, str]]:
    return [event async for event in event_stream(agent, {}, CONFIG)]


@pytest.mark.anyio
async def test_event_stream_emits_token_message_done(mocker: MockerFixture) -> None:
    agent = make_agent(
        mocker,
        stream_items=[messages_chunk('Olá')],
        state_messages=[AIMessage('Olá, tudo bem?')],
    )

    events = await _collect(agent)

    assert events == [
        {'event': 'token', 'data': 'Olá'},
        {'event': 'message', 'data': 'Olá, tudo bem?'},
        {'event': 'done', 'data': ''},
    ]


def test_is_streamable_skips_products_agent_internals() -> None:
    chunk = AIMessageChunk(content='pensando...')
    # Raciocínio interno do ReAct agent de produtos (identificado pelo namespace).
    meta = {
        'langgraph_node': 'model',
        'langgraph_checkpoint_ns': f'{Nodes.PRODUCTS}:1|model:2',
    }
    assert _is_streamable_token(chunk, meta) is False


def test_is_streamable_skips_products_node() -> None:
    chunk = AIMessageChunk(content='Que produto você procura?')
    meta = {'langgraph_node': Nodes.PRODUCTS, 'langgraph_checkpoint_ns': ''}
    assert _is_streamable_token(chunk, meta) is False


def test_is_streamable_allows_regular_agent_tokens() -> None:
    chunk = AIMessageChunk(content='Olá')
    meta = {'langgraph_node': Nodes.GUIDE, 'langgraph_checkpoint_ns': 'guide:1'}
    assert _is_streamable_token(chunk, meta) is True


@pytest.mark.anyio
async def test_event_stream_skips_supervisor_tokens(mocker: MockerFixture) -> None:
    agent = make_agent(
        mocker,
        stream_items=[messages_chunk('{"next": "guide"}', node=Nodes.SUPERVISOR)],
        state_messages=[AIMessage('resposta final')],
    )

    events = await _collect(agent)

    assert {'event': 'token', 'data': '{"next": "guide"}'} not in events
    assert events == [
        {'event': 'message', 'data': 'resposta final'},
        {'event': 'done', 'data': ''},
    ]


@pytest.mark.anyio
async def test_event_stream_skips_empty_chunks(mocker: MockerFixture) -> None:
    agent = make_agent(
        mocker,
        stream_items=[messages_chunk('')],
        state_messages=[AIMessage('final')],
    )

    events = await _collect(agent)

    assert all(event['event'] != 'token' for event in events)


@pytest.mark.anyio
async def test_event_stream_only_done_when_no_final_message(
    mocker: MockerFixture,
) -> None:
    agent = make_agent(mocker, stream_items=[], state_messages=[])

    events = await _collect(agent)

    assert events == [{'event': 'done', 'data': ''}]
