from decimal import Decimal
from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from pytest_mock import MockerFixture

from app.core.agents.constants import Phase
from app.core.agents.products import nodes as products_nodes
from app.core.agents.products.constants import Nodes
from app.core.agents.products.nodes import (
    MAX_SEARCH_ATTEMPTS,
    TOP_PRODUCTS,
    collect_requirements_node,
    execute_search_node,
    plan_search_node,
    present_recommendations_node,
    route_after_collect,
    route_after_validate,
    validate_products_node,
)
from app.core.agents.products.schemas import (
    CollectedInfo,
    Product,
    PurchaseLink,
    Requirements,
    ReviewVerdict,
    SearchPlan,
)
from app.core.agents.products.state import (
    ProductDict,
    ProductSearchState,
    RequirementsDict,
)
from tests.utils import patch_structured_llm


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_product(name: str = 'Galaxy A17', price: str | None = '1500.00') -> Product:
    return Product(
        name=name,
        reason='bom custo-benefício',
        estimated_price=Decimal(price) if price is not None else None,
    )


def product_dict(
    name: str = 'Galaxy A17', price: str | None = '1500.00'
) -> ProductDict:
    return cast(ProductDict, make_product(name, price).model_dump(mode='json'))


BUDGET = 2000.0
COMPLETE_REQUIREMENTS: RequirementsDict = {
    'product_type': 'celular',
    'use_case': 'fotos',
    'priorities': ['câmera'],
    'brand_preferences': [],
    'must_haves': [],
}


# --------------------------------------------------------------------------- #
# collect_requirements_node (next-best-question, turn-based)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_collect_incomplete_asks_and_sets_phase(
    mocker: MockerFixture,
) -> None:
    llm, structured_ainvoke, _ = patch_structured_llm(
        mocker,
        products_nodes,
        structured_return=CollectedInfo(),  # nada coletado -> incompleto
    )
    llm.ainvoke.return_value = AIMessage('Que tipo de produto você procura?')

    state: ProductSearchState = {  # type: ignore
        'messages': [HumanMessage('oi')],
    }
    result = await collect_requirements_node(state)

    assert result['phase'] == Phase.PRODUCTS_COLLECTING.value
    assert isinstance(result['messages'][0], AIMessage)
    assert result['messages'][0].content == 'Que tipo de produto você procura?'
    structured_ainvoke.assert_awaited_once()
    llm.ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_collect_asks_budget_when_only_budget_missing(
    mocker: MockerFixture,
) -> None:
    llm, _, _ = patch_structured_llm(
        mocker,
        products_nodes,
        structured_return=CollectedInfo(
            product_type='celular', use_case='fotos', budget=None
        ),
    )
    llm.ainvoke.return_value = AIMessage('Qual seu orçamento máximo?')

    state: ProductSearchState = {  # type: ignore
        'messages': [HumanMessage('quero um celular pra fotos')],
    }
    result = await collect_requirements_node(state)

    assert result['phase'] == Phase.PRODUCTS_COLLECTING.value
    assert result['messages'][0].content == 'Qual seu orçamento máximo?'
    # A pergunta gerada é sobre o orçamento (a lacuna de maior valor restante).
    sent = llm.ainvoke.call_args.args[0]
    assert 'orçamento' in sent[0].content


@pytest.mark.anyio
async def test_collect_complete_clears_phase_and_no_message(
    mocker: MockerFixture,
) -> None:
    llm, structured_ainvoke, _ = patch_structured_llm(
        mocker,
        products_nodes,
        structured_return=CollectedInfo(
            product_type='celular', use_case='fotos', budget=2000.0
        ),
    )

    state: ProductSearchState = {  # type: ignore
        'messages': [HumanMessage('quero um celular pra fotos até 2000')],
    }
    result = await collect_requirements_node(state)

    assert result['phase'] == ''
    assert result['budget'] == BUDGET
    assert 'messages' not in result
    structured_ainvoke.assert_awaited_once()
    llm.ainvoke.assert_not_awaited()


def test_route_after_collect_phase_active_ends_turn() -> None:
    state: ProductSearchState = {  # type: ignore
        'phase': Phase.PRODUCTS_COLLECTING.value,
    }
    assert route_after_collect(state) == END


def test_route_after_collect_complete_goes_to_plan() -> None:
    state: ProductSearchState = {  # type: ignore
        'phase': '',
    }
    assert route_after_collect(state) == Nodes.PLAN


# --------------------------------------------------------------------------- #
# plan_search_node
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_plan_first_attempt_no_replan_hint(mocker: MockerFixture) -> None:
    _, structured_ainvoke, _ = patch_structured_llm(
        mocker,
        products_nodes,
        structured_return=SearchPlan(queries=['q1', 'q2', 'q3']),
    )

    state: ProductSearchState = {  # type: ignore
        'requirements': COMPLETE_REQUIREMENTS,
        'budget': BUDGET,
    }
    result = await plan_search_node(state)

    assert result['plan'] == ['q1', 'q2', 'q3']
    human_message = structured_ainvoke.call_args.args[0][1]
    assert 'replanejamento' not in human_message.content


@pytest.mark.anyio
async def test_plan_replan_includes_hint(mocker: MockerFixture) -> None:
    _, structured_ainvoke, _ = patch_structured_llm(
        mocker,
        products_nodes,
        structured_return=SearchPlan(queries=['q1']),
    )

    state: ProductSearchState = {  # type: ignore
        'requirements': COMPLETE_REQUIREMENTS,
        'budget': BUDGET,
        'search_attempts': 1,
    }
    await plan_search_node(state)

    human_message = structured_ainvoke.call_args.args[0][1]
    assert 'replanejamento' in human_message.content


# --------------------------------------------------------------------------- #
# execute_search_node
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_execute_runs_planned_searches_and_extracts(
    mocker: MockerFixture,
) -> None:
    run_searches = mocker.patch.object(
        products_nodes,
        'run_planned_searches',
        new=mocker.AsyncMock(return_value=['r1', 'r2']),
    )
    extract = mocker.patch.object(
        products_nodes,
        'extract_candidates',
        new=mocker.AsyncMock(return_value=[make_product()]),
    )

    state: ProductSearchState = {  # type: ignore
        'requirements': COMPLETE_REQUIREMENTS,
        'budget': BUDGET,
        'plan': ['q1', 'q2'],
        'search_attempts': 0,
    }
    result = await execute_search_node(state)

    assert result['search_attempts'] == 1
    assert len(result['products']) == 1
    run_searches.assert_awaited_once_with(['q1', 'q2'])
    extract.assert_awaited_once()
    # Requirements reconstruídos e budget repassados para a extração.
    assert isinstance(extract.call_args.args[1], Requirements)
    assert extract.call_args.args[2] == BUDGET


# --------------------------------------------------------------------------- #
# validate_products_node (orçamento + avaliações online)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_validate_filters_over_budget_and_keeps_well_rated(
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        products_nodes,
        '_review_verdict',
        new=mocker.AsyncMock(
            return_value=ReviewVerdict(well_rated=True, summary='nota 4.5')
        ),
    )

    state: ProductSearchState = {  # type: ignore
        'products': [
            product_dict('Barato', '1500.00'),
            product_dict('Caro', '2500.00'),  # fora do orçamento -> descartado
            product_dict('SemPreco', None),
        ],
        'budget': 2000.0,
    }
    result = await validate_products_node(state)

    names = [p['name'] for p in result['products']]
    assert names == ['Barato', 'SemPreco']
    # O resumo da avaliação é anexado aos que passam.
    assert all(p['review_summary'] == 'nota 4.5' for p in result['products'])


@pytest.mark.anyio
async def test_validate_drops_poorly_rated(mocker: MockerFixture) -> None:
    verdicts = [
        ReviewVerdict(well_rated=True, summary='ótimo'),
        ReviewVerdict(well_rated=False, summary='muitas reclamações'),
    ]
    mocker.patch.object(
        products_nodes,
        '_review_verdict',
        new=mocker.AsyncMock(side_effect=verdicts),
    )

    state: ProductSearchState = {  # type: ignore
        'products': [product_dict('Bom', '1000.00'), product_dict('Ruim', '1000.00')],
        'budget': None,
    }
    result = await validate_products_node(state)

    names = [p['name'] for p in result['products']]
    assert names == ['Bom']


# --------------------------------------------------------------------------- #
# present_recommendations_node (com links anexados)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_present_attaches_links_and_formats(mocker: MockerFixture) -> None:
    links = [PurchaseLink(store='Amazon', url='https://amazon.com/x', price=1500.0)]
    fetch = mocker.patch.object(
        products_nodes,
        'fetch_purchase_links',
        new=mocker.AsyncMock(return_value=links),
    )

    products = [product_dict('Primeiro'), product_dict('Segundo')]
    state: ProductSearchState = {  # type: ignore
        'products': products,
    }
    result = await present_recommendations_node(state)

    message = result['messages'][0]
    assert isinstance(message, AIMessage)
    assert 'Primeiro' in message.content
    assert 'Amazon' in message.content
    assert 'https://amazon.com/x' in message.content
    # Links buscados por produto e persistidos no estado.
    assert fetch.await_count == len(products)
    assert result['products'][0]['purchase_links'][0]['store'] == 'Amazon'


# --------------------------------------------------------------------------- #
# route_after_validate
# --------------------------------------------------------------------------- #
def test_route_after_validate_enough_products_presents() -> None:
    state: ProductSearchState = {  # type: ignore
        'products': [product_dict(f'P{i}') for i in range(TOP_PRODUCTS)],
        'search_attempts': 1,
    }
    assert route_after_validate(state) == Nodes.PRESENT


def test_route_after_validate_max_attempts_presents() -> None:
    state: ProductSearchState = {  # type: ignore
        'products': [],
        'search_attempts': MAX_SEARCH_ATTEMPTS,
    }
    assert route_after_validate(state) == Nodes.PRESENT


def test_route_after_validate_otherwise_replans() -> None:
    state: ProductSearchState = {  # type: ignore
        'products': [product_dict('Único')],
        'search_attempts': 1,
    }
    assert route_after_validate(state) == Nodes.PLAN
