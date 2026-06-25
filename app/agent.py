from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from app.config import Settings
from app.tools import add, multiply, subtract


def build_agent():
    settings = Settings()  # type: ignore[call-args]

    llm = init_chat_model(settings.AGENT_CHAT_MODEL)
    agent = create_agent(
        model=llm,
        tools=[multiply, add, subtract],
        system_prompt='Você é um assistente prestativo',
    )

    return agent
