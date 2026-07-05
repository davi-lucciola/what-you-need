from app.core.agents.products import build_product_search_node
from app.core.agents.products.agent import product_agent_node


def test_build_product_search_node_returns_the_agent_node() -> None:
    # O nó de produtos agora é uma única função (ReAct agent), não um subgrafo.
    assert build_product_search_node() is product_agent_node
