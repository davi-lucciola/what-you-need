from typing import TypedDict

from langchain.agents import AgentState


# Formato JSON-native dos modelos guardados no estado do sub-grafo. Espelham o
# `model_dump(mode='json')` dos respectivos modelos pydantic (Decimal vira string no
# modo json). Guardar dicts tipados — e não objetos pydantic — evita que o checkpointer
# tenha de desserializar tipos "não registrados" (msgpack), o que será bloqueado no futuro.
class RequirementsDict(TypedDict):
    product_type: str | None
    use_case: str | None
    priorities: list[str]
    brand_preferences: list[str]
    must_haves: list[str]


class ProductDict(TypedDict):
    name: str
    brand: str | None
    estimated_price: str | None  # Decimal serializado como string (mode='json').
    reason: str
    key_features: list[str]


class ProductSearchState(AgentState):
    requirements: RequirementsDict | None
    budget: float | None
    # Pergunta já gerada e exibida ao usuário, aguardando resposta. Persistida
    # antes do interrupt() para garantir fidelidade: como o nó re-executa do topo
    # no resume, relemos o texto exato exibido em vez de recalculá-lo via LLM.
    pending_question: str | None
    products: list[ProductDict]
    chosen_product: ProductDict | None
    search_attempts: int
