from langchain_core.messages import SystemMessage

from app.agents.guide.prompt import GUIDE_SYSTEM_PROMPT
from app.agents.states import ChatState
from app.llm import get_llm


async def build_guide_agent(state: ChatState) -> ChatState:
    llm = get_llm()

    system_message = SystemMessage(GUIDE_SYSTEM_PROMPT)
    messages = [system_message, *state['messages']]
    ai_message = await llm.ainvoke(messages)

    return {'messages': [ai_message]}
