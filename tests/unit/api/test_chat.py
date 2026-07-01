import json
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command
from pytest_mock import MockerFixture
from sse_starlette.sse import EventSourceResponse

from app.agents.constants import Nodes
from app.api import chat as chat_module
from app.api.chat import (
    ChatRequest,
    _event_stream,  # pyright: ignore[reportPrivateUsage]
    _final_message,  # pyright: ignore[reportPrivateUsage]
    _interrupt_value,  # pyright: ignore[reportPrivateUsage]
    chat,
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
    interrupts: list[Any] | None = None,
) -> Any:
    """Monta um agente compilado mockado.

    - `agent.astream(...)` é chamado (não awaited) e iterado com `async for`, então
      retorna um async iterator via `_aiter`.
    - `agent.aget_state(...)` é awaited e retorna um snapshot com `.values` e
      `.interrupts`.
    """
    snapshot = SimpleNamespace(
        values={'messages': state_messages or []},
        interrupts=interrupts,
    )
    agent = mocker.Mock()
    agent.astream = mocker.Mock(return_value=_aiter(stream_items or []))
    agent.aget_state = mocker.AsyncMock(return_value=snapshot)
    return agent


def messages_chunk(text: str, node: str = Nodes.GUIDE) -> Any:
    """Item de stream no modo `'messages'`: `(mode, (chunk, metadata))`."""
    return 'messages', (AIMessageChunk(content=text), {'langgraph_node': node})


def interrupt_update(value: Any) -> Any:
    """Item de stream no modo `'updates'` carregando um interrupt."""
    return 'updates', {'__interrupt__': [SimpleNamespace(value=value)]}


# --------------------------------------------------------------------------- #
# _interrupt_value (função pura, síncrona)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ('mode', 'payload', 'expected'),
    [
        ('updates', {'__interrupt__': [SimpleNamespace(value={'q': 'x'})]}, {'q': 'x'}),
        ('updates', {'__interrupt__': []}, None),
        ('updates', {'other': 1}, None),
        ('messages', {'__interrupt__': [SimpleNamespace(value='x')]}, None),
        ('updates', ('not', 'a', 'dict'), None),
    ],
)
def test_interrupt_value(mode: str, payload: Any, expected: Any) -> None:
    assert _interrupt_value(mode, payload) == expected


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
# _event_stream (async generator — núcleo)
# --------------------------------------------------------------------------- #
async def _collect(agent: Any) -> list[dict[str, str]]:
    return [event async for event in _event_stream(agent, {}, CONFIG)]


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
async def test_event_stream_interrupt_ends_early(mocker: MockerFixture) -> None:
    question = {'type': 'collect', 'question': 'Qual seu orçamento?'}
    agent = make_agent(
        mocker,
        stream_items=[
            interrupt_update(question),
            messages_chunk('não deveria ser lido'),
        ],
        state_messages=[AIMessage('não deveria virar message')],
    )

    events = await _collect(agent)

    assert [event['event'] for event in events] == ['interrupt', 'done']
    assert json.loads(events[0]['data']) == question
    assert events[1] == {'event': 'done', 'data': ''}
    # Interrupt encerra o turno sem buscar a mensagem final autoritativa.
    agent.aget_state.assert_not_awaited()


@pytest.mark.anyio
async def test_event_stream_only_done_when_no_final_message(
    mocker: MockerFixture,
) -> None:
    agent = make_agent(mocker, stream_items=[], state_messages=[])

    events = await _collect(agent)

    assert events == [{'event': 'done', 'data': ''}]


# --------------------------------------------------------------------------- #
# chat (endpoint, async)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_chat_new_message_when_no_pending_interrupt(
    mocker: MockerFixture,
) -> None:
    agent = make_agent(mocker, interrupts=None)
    stream = mocker.patch.object(chat_module, '_event_stream', return_value=_aiter([]))

    response = await chat('t1', ChatRequest(message='quero um celular'), agent)

    assert isinstance(response, EventSourceResponse)
    agent.aget_state.assert_awaited_once_with(CONFIG)
    graph_input = stream.call_args.args[1]
    assert graph_input['next'] == ''
    assert isinstance(graph_input['messages'][0], HumanMessage)
    assert graph_input['messages'][0].content == 'quero um celular'
    assert stream.call_args.args[2] == CONFIG


@pytest.mark.anyio
async def test_chat_resumes_when_pending_interrupt(mocker: MockerFixture) -> None:
    agent = make_agent(mocker, interrupts=[SimpleNamespace(value='?')])
    stream = mocker.patch.object(chat_module, '_event_stream', return_value=_aiter([]))

    response = await chat('t1', ChatRequest(message='2000 reais'), agent)

    assert isinstance(response, EventSourceResponse)
    graph_input = stream.call_args.args[1]
    assert isinstance(graph_input, Command)
    assert graph_input.resume == '2000 reais'
