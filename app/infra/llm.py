from functools import lru_cache

from langchain.chat_models import init_chat_model

from app.config import Settings


@lru_cache
def get_llm():
    settings = Settings()  # type: ignore
    return init_chat_model(settings.AGENT_CHAT_MODEL)
