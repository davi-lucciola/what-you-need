from app.agents.products.agent import build_product_search_agent
from app.agents.products.constants import Nodes


def test_compiles_subgraph_with_all_nodes() -> None:
    graph = build_product_search_agent()

    node_names = set(graph.get_graph().nodes)

    for node in Nodes:
        assert node.value in node_names

    assert '__start__' in node_names
    assert '__end__' in node_names
