import asyncio
from typing import cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.core.agents.products.prompt import (
    FIND_PRODUCTS_PROMPT,
    FIND_PURCHASE_LINKS_PROMPT,
)
from app.core.agents.products.schemas import (
    Product,
    ProductRecommendations,
    PurchaseLink,
    PurchaseLinks,
    Requirements,
)
from app.infra.llm import get_llm
from app.infra.tavily import (
    TavilyResult,
    TavilySearchResponse,
    get_tavily_client,
)

# Modelos pequenos ocasionalmente emitem JSON malformado na saída estruturada;
# tentar novamente com uma nova amostragem resolve a maioria desses casos.
_STRUCTURED_RETRY_ERRORS = (ValidationError, OutputParserException, ValueError)

# Limite de candidatos extraídos por busca e de resultados agregados enviados ao LLM.
_MAX_AGGREGATED_RESULTS = 12


def _format_results(results: list[TavilyResult]) -> str:
    """Monta um contexto textual a partir dos resultados do Tavily (para o LLM)."""
    blocks: list[str] = []
    for result in results:
        blocks.append(
            f'- Título: {result["title"]}\n'
            f'  URL: {result["url"]}\n'
            f'  Relevância: {result["score"]}\n'
            f'  Conteúdo: {result["content"]}'
        )
    return '\n'.join(blocks)


def _dedupe_results(results: list[TavilyResult]) -> list[TavilyResult]:
    """Remove resultados duplicados por URL, preservando a ordem de chegada."""
    seen: set[str] = set()
    unique: list[TavilyResult] = []
    for result in results:
        url = result.get('url', '')
        if url and url not in seen:
            seen.add(url)
            unique.append(result)
    return unique


async def run_planned_searches(queries: list[str]) -> list[TavilyResult]:
    """Executa (concorrentemente) as queries do plano e agrega/dedupe os resultados."""
    client = get_tavily_client()
    responses = await asyncio.gather(
        *(
            client.search(query, search_depth='advanced', country='brazil')
            for query in queries
        )
    )

    aggregated: list[TavilyResult] = []
    for response in responses:
        aggregated.extend(cast(TavilySearchResponse, response).get('results', []))

    return _dedupe_results(aggregated)[:_MAX_AGGREGATED_RESULTS]


async def extract_candidates(
    results: list[TavilyResult], requirements: Requirements, budget: float | None
) -> list[Product]:
    """Extrai os produtos de melhor custo-benefício a partir dos resultados de busca."""
    context = _format_results(results)

    llm = (
        get_llm()
        .with_structured_output(ProductRecommendations, method='function_calling')
        .with_retry(
            retry_if_exception_type=_STRUCTURED_RETRY_ERRORS, stop_after_attempt=3
        )
    )
    priorities = ', '.join(requirements.priorities) or 'não informadas'
    brands = ', '.join(requirements.brand_preferences) or 'indiferente'
    must_haves = ', '.join(requirements.must_haves) or 'nenhum'
    budget_text = f'R${budget}' if budget else 'não informado'
    recommendations = await llm.ainvoke(
        [
            SystemMessage(FIND_PRODUCTS_PROMPT),
            HumanMessage(
                f'Tipo de produto: {requirements.product_type}\n'
                f'Uso pretendido: {requirements.use_case}\n'
                f'Prioridades: {priorities}\n'
                f'Marcas preferidas: {brands}\n'
                f'Requisitos obrigatórios: {must_haves}\n'
                f'Orçamento máximo: {budget_text}\n\n'
                f'Resultados de busca:\n{context}'
            ),
        ]
    )

    assert isinstance(recommendations, ProductRecommendations)
    return recommendations.products


async def search_product_reviews(product: Product) -> str:
    """Busca avaliações/reviews de um produto e devolve o contexto textual.

    Usado na validação para o LLM julgar se o produto é bem avaliado.
    """
    client = get_tavily_client()
    brand = f' {product.brand}' if product.brand else ''
    response = cast(
        TavilySearchResponse,
        await client.search(
            f'{product.name}{brand} avaliações reviews é bom vale a pena',
            search_depth='advanced',
            country='brazil',
        ),
    )
    return _format_results(response.get('results', []))


async def fetch_purchase_links(
    product: Product, quantity: int = 2
) -> list[PurchaseLink]:
    """Busca links de compra para um produto (versão enxuta, sem extract pesado).

    Roda uma única busca por loja e extrai link + preço via LLM a partir dos snippets.
    Diferente do fluxo antigo (um produto escolhido), aqui é chamada para vários
    produtos em paralelo, então evita o `extract` do Tavily para controlar créditos.
    """
    client = get_tavily_client()
    response = cast(
        TavilySearchResponse,
        await client.search(
            f'comprar {product.name} preço Amazon Mercado Livre Shopee',
            search_depth='advanced',
            country='brazil',
        ),
    )
    context = _format_results(response.get('results', []))

    system_message = FIND_PURCHASE_LINKS_PROMPT.format(quantity=quantity)
    human_message = HumanMessage(
        f'Produto: {product.name}'
        f'{f" ({product.brand})" if product.brand else ""}\n\n'
        f'Resultados de busca:\n{context}'
    )
    llm = (
        get_llm()
        .with_structured_output(PurchaseLinks, method='function_calling')
        .with_retry(
            retry_if_exception_type=_STRUCTURED_RETRY_ERRORS, stop_after_attempt=3
        )
    )
    purchase_links = await llm.ainvoke([system_message, human_message])

    assert isinstance(purchase_links, PurchaseLinks)
    return purchase_links.links
