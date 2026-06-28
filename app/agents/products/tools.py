from typing import cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.agents.products.prompt import (
    FIND_PRODUCTS_PROMPT,
    FIND_PURCHASE_LINKS_PROMPT,
)
from app.agents.products.schemas import (
    Product,
    ProductRecommendations,
    PurchaseLink,
    PurchaseLinks,
    Requirements,
)
from app.llm import get_llm
from app.tavily import (
    TavilyExtractResponse,
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
    """Monta um contexto textual a partir dos resultados do Tavily.

    Usado para extração via LLM.
    """
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


def _build_search_queries(
    requirements: Requirements, budget: float | None, refine_hint: str | None = None
) -> list[str]:
    """Gera múltiplas queries direcionadas a partir dos requisitos do usuário.

    Diferentes ângulos (custo-benefício, reviews/comparativos, prioridades) aumentam a
    chance de cobrir os melhores modelos do que uma única query fixa.
    """
    product = requirements.product_type or 'produto'
    budget_clause = f' até R${budget:.0f}' if budget else ''
    priorities = ' '.join(requirements.priorities)
    use_case = requirements.use_case or ''

    queries = [
        f'melhores {product} custo-benefício{budget_clause} 2026',
        f'{product} {use_case} {priorities} review comparativo 2026'.strip(),
    ]
    if priorities:
        queries.append(f'melhor {product} para {priorities}{budget_clause}')
    if requirements.must_haves:
        queries.append(
            f'{product} com {" ".join(requirements.must_haves)}{budget_clause}'
        )
    if refine_hint:
        queries.append(refine_hint)

    # Remove duplicatas mantendo a ordem.
    return list(dict.fromkeys(q for q in queries if q.strip()))


async def search_candidates(
    requirements: Requirements, budget: float | None, refine_hint: str | None = None
) -> list[Product]:
    """Encontra os 3 modelos de melhor custo-benefício para os requisitos informados.

    Roda múltiplas buscas direcionadas no Tavily (custo-benefício, reviews/comparativos,
    prioridades), agrega e deduplica os resultados, e extrai os candidatos via LLM.
    `refine_hint` é usado no loop de re-busca quando os resultados anteriores
    foram fracos.
    """
    client = get_tavily_client()
    queries = _build_search_queries(requirements, budget, refine_hint)

    aggregated: list[TavilyResult] = []
    for query in queries:
        response = cast(
            TavilySearchResponse,
            await client.search(query, search_depth='advanced', country='brazil'),
        )
        aggregated.extend(response.get('results', []))

    results = _dedupe_results(aggregated)[:_MAX_AGGREGATED_RESULTS]
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


async def deep_search_purchase_links(
    product: Product, quantity: int = 2
) -> list[PurchaseLink]:
    """Pesquisa profunda de links de compra para um produto já escolhido pelo usuário.

    Primeiro faz uma busca avançada para localizar páginas de lojas e, em
    seguida, usa o `extract` do Tavily para ler o conteúdo dessas páginas e
    extrair link + preço com mais precisão do que apenas os snippets de busca.
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

    results = response.get('results', [])
    context = _format_results(results)

    # Lê as páginas das lojas mais relevantes para confirmar link e preço reais.
    top_urls = [r['url'] for r in results[:quantity] if r.get('url')]

    if top_urls:
        try:
            extracted = cast(TavilyExtractResponse, await client.extract(top_urls))
            for item in extracted.get('results', []):
                raw = item.get('raw_content') or ''
                if raw:
                    context += (
                        f'\n\n[Conteúdo extraído de {item.get("url", "")}]:\n'
                        f'{raw[:2000]}'
                    )
        except Exception:
            # Se o extract falhar, seguimos apenas com os snippets da busca.
            pass

    # Alternativa mais robusta (e mais cara em créditos): client.research(...).

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
