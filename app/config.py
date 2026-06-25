from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = Field()
    AGENT_CHAT_MODEL: str = Field('google_genai:gemini-2.5-flash')
