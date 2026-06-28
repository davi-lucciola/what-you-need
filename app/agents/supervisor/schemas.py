from pydantic import BaseModel, Field

from app.agents.constants import Agents


class Router(BaseModel):
    next: Agents = Field(
        description=(
            'O agente que o supervisor irá rotear, '
            'pode ser um dos valores do enum "Agents"'
        )
    )
