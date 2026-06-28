import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pytest_mock import MockerFixture

from app.agents.guide import agent as guide_agent
from app.agents.guide.agent import build_guide_agent
from app.agents.guide.prompt import GUIDE_SYSTEM_PROMPT
from app.agents.states import ChatState
from tests.utils import patch_structured_llm


@pytest.mark.anyio
async def test_returns_ai_message_from_llm(mocker: MockerFixture) -> None:
    ai_message = AIMessage('Olá! Como posso ajudar?')
    llm, _, _ = patch_structured_llm(mocker, guide_agent)
    llm.ainvoke.return_value = ai_message

    state: ChatState = {'messages': [HumanMessage('Oii')], 'next': ''}
    result = await build_guide_agent(state)

    assert result == {'messages': [ai_message]}
    llm.ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_prepends_system_prompt_before_state_messages(
    mocker: MockerFixture,
) -> None:
    llm, _, _ = patch_structured_llm(mocker, guide_agent)
    llm.ainvoke.return_value = AIMessage('oi')

    human_message = HumanMessage('Oii')
    state: ChatState = {'messages': [human_message], 'next': ''}
    await build_guide_agent(state)

    llm.ainvoke.assert_awaited_once()
    sent_messages = llm.ainvoke.call_args.args[0]
    system_message = sent_messages[0]
    assert isinstance(system_message, SystemMessage)
    assert system_message.content == GUIDE_SYSTEM_PROMPT
    assert sent_messages[1:] == [human_message]
