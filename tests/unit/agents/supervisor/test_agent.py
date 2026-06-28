import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pytest_mock import MockerFixture

from app.agents.constants import Agents
from app.agents.states import ChatState
from app.agents.supervisor import agent as supervisor_agent
from app.agents.supervisor.agent import build_supervisor_agent
from app.agents.supervisor.prompt import SUPERVISOR_SYSTEM_PROMPT
from app.agents.supervisor.schemas import Router
from tests.utils import patch_structured_llm


@pytest.mark.anyio
@pytest.mark.parametrize('expected_agent', list(Agents))
async def test_routes_to_agent_and_returns_plain_string(
    mocker: MockerFixture,
    expected_agent: Agents,
) -> None:
    _, ainvoke, _ = patch_structured_llm(
        mocker, supervisor_agent, structured_return=Router(next=expected_agent)
    )

    state: ChatState = {'messages': [HumanMessage('Oii')], 'next': ''}
    result = await build_supervisor_agent(state)

    assert result == {'next': expected_agent.value, 'messages': []}
    assert isinstance(result.get('next', ''), str)
    ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_prepends_system_prompt_before_state_messages(
    mocker: MockerFixture,
) -> None:
    _, ainvoke, _ = patch_structured_llm(
        mocker, supervisor_agent, structured_return=Router(next=Agents.GUIDE)
    )

    human_message = HumanMessage('Oii')
    state: ChatState = {'messages': [human_message], 'next': ''}
    await build_supervisor_agent(state)

    ainvoke.assert_awaited_once()
    sent_messages = ainvoke.call_args.args[0]
    system_message = sent_messages[0]
    assert isinstance(system_message, SystemMessage)
    assert system_message.content == SUPERVISOR_SYSTEM_PROMPT
    assert sent_messages[1:] == [human_message]


@pytest.mark.anyio
async def test_uses_router_structured_output(
    mocker: MockerFixture,
) -> None:
    _, ainvoke, with_structured_output = patch_structured_llm(
        mocker, supervisor_agent, structured_return=Router(next=Agents.GUIDE)
    )

    state: ChatState = {'messages': [HumanMessage('Oii')], 'next': ''}
    await build_supervisor_agent(state)

    with_structured_output.assert_called_once_with(Router)
    ainvoke.assert_awaited_once()
