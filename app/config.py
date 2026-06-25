from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AGENT_CHAT_MODEL: str
