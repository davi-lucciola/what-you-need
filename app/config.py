from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AGENT_CHAT_MODEL: str
    TAVILY_API_KEY: str

    CHECKPOINTER_DATABASE_URL: str
