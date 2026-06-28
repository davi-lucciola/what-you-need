from decimal import Decimal
from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pytest_mock import MockerFixture

from app.agents.products import nodes as products_nodes
from app.agents.products.constants import Nodes
from app.agents.products.nodes import (
    MAX_SEARCH_ATTEMPTS,
    TOP_PRODUCTS,
    collect_requirements_node,
    present_recommendations_node,
    route_after_collect,
    route_after_validate,
    search_products_node,
    search_purchase_links_node,
    validate_products_node,
)
from app.agents.products.schemas import (
    CollectedInfo,
    Product,
    ProductChoice,
    PurchaseLink,
    Requirements,
)
from app.agents.products.state import ProductDict, ProductSearchState, RequirementsDict
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
# collect_requirements_node
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_collect_ask_pass_incomplete_returns_question(
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

    assert result['pending_question'] == 'Que tipo de produto você procura?'
    assert isinstance(result['messages'][0], AIMessage)
    assert result['messages'][0].content == 'Que tipo de produto você procura?'
    structured_ainvoke.assert_awaited_once()
    llm.ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_collect_ask_pass_complete_has_no_question(
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

    assert result['pending_question'] is None
    assert result['budget'] == BUDGET
    assert 'messages' not in result
    structured_ainvoke.assert_awaited_once()
    llm.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_collect_collect_pass_interrupts_and_extracts(
    mocker: MockerFixture,
) -> None:
    _, structured_ainvoke, _ = patch_structured_llm(
        mocker,
        products_nodes,
        structured_return=CollectedInfo(
            product_type='celular', use_case='fotos', budget=2000.0
        ),
    )
    interrupt = mocker.patch.object(
        products_nodes, 'interrupt', return_value='2000 reais'
    )

    pending = 'Qual seu orçamento máximo?'
    state: ProductSearchState = {  # type: ignore
        'messages': [HumanMessage('oi')],
        'pending_question': pending,
    }
    result = await collect_requirements_node(state)

    assert result['pending_question'] is None
    assert result['budget'] == BUDGET
    assert isinstance(result['messages'][0], HumanMessage)
    assert result['messages'][0].content == '2000 reais'
    interrupt.assert_called_once_with(
        {'type': 'collect', 'message': '', 'question': pending}
    )
    structured_ainvoke.assert_awaited_once()


# --------------------------------------------------------------------------- #
# search_products_node
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_search_products_first_attempt_no_refine_hint(
    mocker: MockerFixture,
) -> None:
    search_candidates = mocker.patch.object(
        products_nodes,
        'search_candidates',
        new=mocker.AsyncMock(return_value=[make_product()]),
    )

    state: ProductSearchState = {  # type: ignore
        'requirements': COMPLETE_REQUIREMENTS,
        'budget': 2000.0,
    }
    result = await search_products_node(state)

    assert result['search_attempts'] == 1
    assert len(result['products']) == 1
    search_candidates.assert_awaited_once()
    call = search_candidates.call_args
    assert isinstance(call.args[0], Requirements)
    assert call.args[0].product_type == 'celular'
    assert call.args[1] == BUDGET
    assert call.args[2] is None


@pytest.mark.anyio
async def test_search_products_reloop_passes_refine_hint(
    mocker: MockerFixture,
) -> None:
    search_candidates = mocker.patch.object(
        products_nodes,
        'search_candidates',
        new=mocker.AsyncMock(return_value=[make_product()]),
    )

    attempts = 1
    state: ProductSearchState = {  # type: ignore
        'requirements': COMPLETE_REQUIREMENTS,
        'budget': BUDGET,
        'search_attempts': attempts,
    }
    result = await search_products_node(state)

    assert result['search_attempts'] == attempts + 1
    search_candidates.assert_awaited_once()
    refine_hint = search_candidates.call_args.args[2]
    assert refine_hint is not None
    assert 'celular' in refine_hint


# --------------------------------------------------------------------------- #
# validate_products_node (lógica pura)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_validate_filters_products_over_budget() -> None:
    state: ProductSearchState = {  # type: ignore
        'products': [
            product_dict('Barato', '1500.00'),
            product_dict('Caro', '2500.00'),
            product_dict('SemPreco', None),
        ],
        'budget': 2000.0,
    }
    result = await validate_products_node(state)

    names = [p['name'] for p in result['products']]
    assert names == ['Barato', 'SemPreco']


@pytest.mark.anyio
async def test_validate_keeps_all_without_budget() -> None:
    products = [product_dict('A', '1500.00'), product_dict('B', '9999.00')]
    state: ProductSearchState = {  # type: ignore
        'products': products,
        'budget': None,
    }
    result = await validate_products_node(state)

    assert len(result['products']) == len(products)


# --------------------------------------------------------------------------- #
# present_recommendations_node
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_present_resolves_user_choice(mocker: MockerFixture) -> None:
    _, structured_ainvoke, _ = patch_structured_llm(
        mocker, products_nodes, structured_return=ProductChoice(index=2)
    )
    interrupt = mocker.patch.object(
        products_nodes, 'interrupt', return_value='o segundo'
    )

    products = [
        product_dict('Primeiro', '1000.00'),
        product_dict('Segundo', '1500.00'),
        product_dict('Terceiro', '1800.00'),
    ]
    state: ProductSearchState = {  # type: ignore
        'products': products,
    }
    result = await present_recommendations_node(state)

    assert result['chosen_product']['name'] == 'Segundo'
    assert isinstance(result['messages'][0], AIMessage)
    assert isinstance(result['messages'][1], HumanMessage)
    assert result['messages'][1].content == 'o segundo'
    interrupt.assert_called_once()
    structured_ainvoke.assert_awaited_once()


# --------------------------------------------------------------------------- #
# search_purchase_links_node
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_purchase_links_formats_found_links(mocker: MockerFixture) -> None:
    links = [PurchaseLink(store='Amazon', url='https://amazon.com/x', price=1500.0)]
    deep_search = mocker.patch.object(
        products_nodes,
        'deep_search_purchase_links',
        new=mocker.AsyncMock(return_value=links),
    )

    state: ProductSearchState = {  # type: ignore
        'chosen_product': product_dict('Galaxy A17'),
    }
    result = await search_purchase_links_node(state)

    message = result['messages'][0]
    assert isinstance(message, AIMessage)
    assert 'Onde comprar o Galaxy A17' in message.content
    assert 'Amazon' in message.content
    deep_search.assert_awaited_once()
    assert isinstance(deep_search.call_args.args[0], Product)
    assert deep_search.call_args.args[0].name == 'Galaxy A17'


@pytest.mark.anyio
async def test_purchase_links_handles_empty_results(mocker: MockerFixture) -> None:
    deep_search = mocker.patch.object(
        products_nodes,
        'deep_search_purchase_links',
        new=mocker.AsyncMock(return_value=[]),
    )

    state: ProductSearchState = {  # type: ignore
        'chosen_product': product_dict('Galaxy A17'),
    }
    result = await search_purchase_links_node(state)

    assert 'Não encontrei links' in result['messages'][0].content
    deep_search.assert_awaited_once()


# --------------------------------------------------------------------------- #
# Roteadores (síncronos)
# --------------------------------------------------------------------------- #
def test_route_after_collect_pending_question_loops() -> None:
    state: ProductSearchState = {  # type: ignore
        'pending_question': 'Qual seu orçamento?',
    }
    assert route_after_collect(state) == Nodes.COLLECT


def test_route_after_collect_complete_goes_to_search() -> None:
    state: ProductSearchState = {  # type: ignore
        'pending_question': None,
        'requirements': COMPLETE_REQUIREMENTS,
        'budget': 2000.0,
    }
    assert route_after_collect(state) == Nodes.SEARCH


def test_route_after_collect_incomplete_loops() -> None:
    state: ProductSearchState = {  # type: ignore
        'pending_question': None,
        'requirements': None,
        'budget': None,
    }
    assert route_after_collect(state) == Nodes.COLLECT


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


def test_route_after_validate_otherwise_searches_again() -> None:
    state: ProductSearchState = {  # type: ignore
        'products': [product_dict('Único')],
        'search_attempts': 1,
    }
    assert route_after_validate(state) == Nodes.SEARCH
