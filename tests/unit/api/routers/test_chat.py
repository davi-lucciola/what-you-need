from typing import Any, AsyncIterator

import pytest
from langchain_core.messages import HumanMessage
from pytest_mock import MockerFixture
from sse_starlette.sse import EventSourceResponse

from app.api.routers.chat import chat
from app.api.schemas.chat import ChatRequest

CONFIG: Any = {'configurable': {'thread_id': 't1'}}


async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


@pytest.mark.anyio
async def test_chat_always_sends_human_message(mocker: MockerFixture) -> None:
    # Turn-based: sem inspeção de interrupt/resume — toda mensagem entra como
    # HumanMessage e o grafo roda START→END.
    agent = mocker.Mock()
    stream = mocker.patch(
        'app.core.services.chat.event_stream', return_value=_aiter([])
    )

    response = await chat('t1', ChatRequest(message='quero um celular'), agent)

    assert isinstance(response, EventSourceResponse)
    graph_input = stream.call_args.args[1]
    assert graph_input['next'] == ''
    assert isinstance(graph_input['messages'][0], HumanMessage)
    assert graph_input['messages'][0].content == 'quero um celular'
    assert stream.call_args.args[2] == CONFIG
