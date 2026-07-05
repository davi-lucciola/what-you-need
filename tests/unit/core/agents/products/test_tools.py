from decimal import Decimal
from typing import Any

import pytest
from pytest_mock import MockerFixture

from app.core.agents.products import tools as products_tools
from app.core.agents.products.schemas import ListingVerdict, Product, PurchaseLink
from app.core.agents.products.tools import (
    build_product_tools,
    select_final_products,
    verify_listing_available,
)


def make_product(name: str, price: str | None) -> Product:
    return Product(
        name=name,
        reason='r',
        estimated_price=Decimal(price) if price is not None else None,
    )


# --------------------------------------------------------------------------- #
# select_final_products (regra de orçamento + garantia de não-vazio)
# --------------------------------------------------------------------------- #
def test_select_no_budget_returns_top_in_order() -> None:
    candidates = [make_product(f'P{i}', '100.00') for i in range(8)]
    result = select_final_products(candidates, budget=None, top=5)
    assert [p.name for p in result] == ['P0', 'P1', 'P2', 'P3', 'P4']


def test_select_with_budget_has_exactly_one_stretch() -> None:
    budget = 1000.0
    candidates = [
        make_product('A', '800.00'),
        make_product('B', '850.00'),
        make_product('C', '900.00'),
        make_product('E', '1000.00'),  # 4 opções na faixa (<= orçamento)
        make_product('D', '1100.00'),  # acima do orçamento, dentro da tolerância (20%)
        make_product('Stretch', '1150.00'),  # acima, mais cara -> não escolhida
    ]
    result = select_final_products(candidates, budget=budget, top=5)
    names = [p.name for p in result]
    above = [
        p
        for p in result
        if p.estimated_price and float(p.estimated_price) > budget
    ]
    # Exatamente uma opção acima do orçamento (a mais barata acima = 'D' a 1100).
    assert len(above) == 1
    assert above[0].name == 'D'
    # A opção stretch fica por último.
    assert names[-1] == 'D'


def test_select_unknown_price_counts_as_within_budget() -> None:
    candidates = [
        make_product('SemPreco', None),
        make_product('Barato', '500.00'),
    ]
    result = select_final_products(candidates, budget=1000.0, top=5)
    assert {p.name for p in result} == {'SemPreco', 'Barato'}


def test_select_never_empty_fills_to_top() -> None:
    # Só 1 dentro do orçamento e 1 acima: nunca-vazio completa até o disponível.
    candidates = [
        make_product('Dentro', '500.00'),
        make_product('Acima', '5000.00'),
    ]
    result = select_final_products(candidates, budget=1000.0, top=5)
    assert len(result) == len(candidates)
    assert 'Dentro' in {p.name for p in result}


def test_select_empty_candidates_returns_empty() -> None:
    assert select_final_products([], budget=1000.0, top=5) == []


# --------------------------------------------------------------------------- #
# verify_listing_available (Tavily extract + veredito por página)
# --------------------------------------------------------------------------- #
@pytest.mark.anyio
async def test_verify_listing_marks_live_and_failed(mocker: MockerFixture) -> None:
    client = mocker.Mock()
    client.extract = mocker.AsyncMock(
        return_value={
            'results': [
                {'url': 'u-live', 'raw_content': 'comprar agora R$1.500 em estoque'},
            ],
            'failed_results': [{'url': 'u-dead', 'error': '404'}],
        }
    )
    mocker.patch.object(products_tools, 'get_tavily_client', return_value=client)
    mocker.patch.object(
        products_tools,
        '_judge_listing',
        new=mocker.AsyncMock(return_value=ListingVerdict(available=True, price=1500.0)),
    )

    result = await verify_listing_available(['u-live', 'u-dead'])

    assert result['u-live'].available is True
    # URLs que falharam na extração são consideradas indisponíveis.
    assert result['u-dead'].available is False


@pytest.mark.anyio
async def test_verify_listing_empty_urls_no_call(mocker: MockerFixture) -> None:
    client = mocker.Mock()
    client.extract = mocker.AsyncMock()
    mocker.patch.object(products_tools, 'get_tavily_client', return_value=client)

    assert await verify_listing_available([]) == {}
    client.extract.assert_not_awaited()


# --------------------------------------------------------------------------- #
# present_recommendations (tool terminal): enriquece + regra de orçamento + formata
# --------------------------------------------------------------------------- #
def _present_tool() -> Any:
    return next(t for t in build_product_tools() if t.name == 'present_recommendations')


@pytest.mark.anyio
async def test_present_recommendations_enriches_and_applies_budget(
    mocker: MockerFixture,
) -> None:
    async def fake_links(product: Product, *_: Any, **__: Any) -> list[PurchaseLink]:
        return [PurchaseLink(store='Amazon', url=f'u-{product.name}', price=None)]

    async def fake_availability(urls: list[str]) -> dict[str, ListingVerdict]:
        return {u: ListingVerdict(available=True, price=999.0) for u in urls}

    mocker.patch.object(
        products_tools,
        'fetch_purchase_links',
        new=mocker.AsyncMock(side_effect=fake_links),
    )
    mocker.patch.object(
        products_tools,
        'verify_listing_available',
        new=mocker.AsyncMock(side_effect=fake_availability),
    )

    present = _present_tool()
    result = await present.ainvoke(
        {
            'products': [
                {'name': 'Barato', 'reason': 'r', 'estimated_price': 800.0,
                 'review_summary': 'ótimo'},
                {'name': 'Esticado', 'reason': 'r', 'estimated_price': 1100.0,
                 'review_summary': 'bom'},
            ],
            'budget': 1000.0,
        }
    )

    assert 'Barato' in result
    assert 'Esticado' in result
    # Link validado é anexado e a opção acima do orçamento ganha a observação.
    assert 'Onde comprar' in result
    assert 'acima do seu orçamento' in result


@pytest.mark.anyio
async def test_present_recommendations_never_empty_message() -> None:
    present = _present_tool()
    result = await present.ainvoke({'products': [], 'budget': None})
    assert 'afrouxar' in result or 'rever' in result
