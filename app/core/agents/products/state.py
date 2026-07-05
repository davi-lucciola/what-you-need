from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RequirementsDict(TypedDict):
    product_type: str | None
    use_case: str | None
    priorities: list[str]
    brand_preferences: list[str]
    must_haves: list[str]


class PurchaseLinkDict(TypedDict):
    store: str
    url: str
    price: float | None


class ProductDict(TypedDict):
    name: str
    brand: str | None
    estimated_price: str | None
    reason: str
    key_features: list[str]
    review_summary: NotRequired[str | None]
    purchase_links: NotRequired[list[PurchaseLinkDict]]


class ProductSearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Compartilhado com o ChatState pai: sobrescrita simples, permite limpar com ''.
    phase: NotRequired[str]
    requirements: RequirementsDict | None
    budget: float | None
    # Queries geradas pelo planner e consumidas pelo executor (dentro de um turno).
    plan: NotRequired[list[str] | None]
    products: list[ProductDict]
    search_attempts: int
