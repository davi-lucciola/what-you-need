from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RequirementsDict(TypedDict):
    product_type: str | None
    use_case: str | None
    priorities: list[str]
    brand_preferences: list[str]
    must_haves: list[str]


class ProductDict(TypedDict):
    name: str
    brand: str | None
    estimated_price: str | None
    reason: str
    key_features: list[str]


class ProductSearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    requirements: RequirementsDict | None
    budget: float | None
    pending_question: str | None
    products: list[ProductDict]
    chosen_product: ProductDict | None
    search_attempts: int
