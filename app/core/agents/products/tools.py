import asyncio
from typing import cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from app.core.agents.products.constants import (
    BUDGET_STRETCH_TOLERANCE,
    TOP_PRODUCTS,
)
from app.core.agents.products.prompt import (
    FIND_PRODUCTS_PROMPT,
    FIND_PURCHASE_LINKS_PROMPT,
    LISTING_VALIDATION_PROMPT,
    REVIEW_VALIDATION_PROMPT,
)
from app.core.agents.products.schemas import (
    ListingVerdict,
    Product,
    ProductRecommendations,
    PurchaseLink,
    PurchaseLinks,
    ReviewVerdict,
)
from app.infra.llm import get_llm
from app.infra.tavily import (
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
# Trecho máximo do conteúdo de página enviado ao LLM na validação de disponibilidade.
_MAX_EXTRACT_CHARS = 4000


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
    results: list[TavilyResult], context: str, budget: float | None
) -> list[Product]:
    """Extrai os produtos de melhor custo-benefício a partir dos resultados de busca.

    `context` é a necessidade do usuário em texto livre (ex. a query da busca).
    """
    search_context = _format_results(results)

    llm = (
        get_llm()
        .with_structured_output(ProductRecommendations, method='function_calling')
        .with_retry(
            retry_if_exception_type=_STRUCTURED_RETRY_ERRORS, stop_after_attempt=3
        )
    )
    budget_text = f'R${budget}' if budget else 'não informado'
    recommendations = await llm.ainvoke(
        [
            SystemMessage(FIND_PRODUCTS_PROMPT),
            HumanMessage(
                f'Necessidade do usuário: {context}\n'
                f'Orçamento máximo: {budget_text}\n\n'
                f'Resultados de busca:\n{search_context}'
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


async def review_verdict(product: Product) -> ReviewVerdict:
    """Julga a reputação online de um produto a partir de resultados de busca."""
    context = await search_product_reviews(product)
    llm = get_llm().with_structured_output(ReviewVerdict, method='function_calling')
    verdict = await llm.ainvoke(
        [
            SystemMessage(REVIEW_VALIDATION_PROMPT),
            HumanMessage(f'Produto: {product.name}\n\nResultados de busca:\n{context}'),
        ]
    )
    assert isinstance(verdict, ReviewVerdict)
    return verdict


async def _judge_listing(url: str, raw_content: str) -> ListingVerdict:
    """Decide se um anúncio está no ar/disponível a partir do conteúdo da página."""
    if not raw_content.strip():
        return ListingVerdict(available=False)
    llm = get_llm().with_structured_output(ListingVerdict, method='function_calling')
    verdict = await llm.ainvoke(
        [
            SystemMessage(LISTING_VALIDATION_PROMPT),
            HumanMessage(
                f'URL: {url}\n\nConteúdo da página:\n{raw_content[:_MAX_EXTRACT_CHARS]}'
            ),
        ]
    )
    assert isinstance(verdict, ListingVerdict)
    return verdict


async def verify_listing_available(urls: list[str]) -> dict[str, ListingVerdict]:
    """Busca o conteúdo de cada anúncio (Tavily extract) e confirma se está no ar.

    Retorna um mapa url -> veredito (disponibilidade + preço extraído da página).
    URLs que falharam na extração são consideradas indisponíveis.
    """
    if not urls:
        return {}

    client = get_tavily_client()
    response = cast(
        TavilyExtractResponse,
        await client.extract(urls, extract_depth='basic'),
    )

    verdicts: dict[str, ListingVerdict] = {}
    extracted = response.get('results', [])
    judged = await asyncio.gather(
        *(_judge_listing(r['url'], r.get('raw_content') or '') for r in extracted)
    )
    for result, verdict in zip(extracted, judged):
        verdicts[result['url']] = verdict
    for failure in response.get('failed_results', []):
        verdicts.setdefault(failure['url'], ListingVerdict(available=False))
    return verdicts


async def fetch_purchase_links(
    product: Product, quantity: int = 2
) -> list[PurchaseLink]:
    """Busca candidatos a links de compra para um produto (sem extract pesado).

    Roda uma única busca por loja e extrai link + preço via LLM a partir dos snippets.
    A validação de disponibilidade dos links fica a cargo de `verify_listing_available`.
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


def _format_candidates(products: list[Product]) -> str:
    """Resumo compacto dos candidatos para o agente (resumo, não dump)."""
    if not products:
        return 'Nenhum produto encontrado nesta busca. Tente outra query.'
    lines: list[str] = []
    for product in products:
        price = (
            f'R${product.estimated_price}'
            if product.estimated_price is not None
            else 'preço não informado'
        )
        brand = f' ({product.brand})' if product.brand else ''
        lines.append(f'- {product.name}{brand} — {price}: {product.reason}')
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Enriquecimento + formatação (usados pela tool terminal present_recommendations).
# Caveats: anotados quando um produto entra na lista sem passar num filtro, para
# nunca voltar vazio — renderizados junto de `unmet_requirements`.
# --------------------------------------------------------------------------- #
CAVEAT_REVIEWS = 'avaliações não confirmadas'
CAVEAT_NO_LISTING = 'sem link de compra ativo no momento'


async def _enrich_product(product: Product) -> Product:
    """Valida disponibilidade do anúncio e anexa links vivos (concorrente por produto).

    As avaliações são responsabilidade do agente (tool `check_reviews`); aqui apenas
    anotamos um caveat se o `review_summary` não veio preenchido.
    """
    links = await fetch_purchase_links(product)
    availability = await verify_listing_available([link.url for link in links])
    live_links: list[PurchaseLink] = []
    for link in links:
        listing = availability.get(link.url)
        if listing is not None and listing.available:
            price = link.price if link.price is not None else listing.price
            live_links.append(link.model_copy(update={'price': price}))

    caveats = list(product.unmet_requirements)
    if not product.review_summary:
        caveats.append(CAVEAT_REVIEWS)
    if not live_links:
        caveats.append(CAVEAT_NO_LISTING)

    return product.model_copy(
        update={
            'available': bool(live_links),
            'purchase_links': live_links,
            'unmet_requirements': caveats,
        }
    )


def _rank_by_quality(products: list[Product]) -> list[Product]:
    """Produtos "limpos" (disponíveis e com avaliação) primeiro, preservando a ordem.

    Preferimos os que passam nos filtros, mas nunca descartamos ninguém — os demais
    seguem anotados com caveats, garantindo que a lista nunca fique vazia.
    """
    clean = [
        p
        for p in products
        if p.available and CAVEAT_REVIEWS not in p.unmet_requirements
    ]
    degraded = [p for p in products if p not in clean]
    return [*clean, *degraded]


def _format_links(links: list[PurchaseLink]) -> str:
    if not links:
        return '   Onde comprar: não encontrei links confiáveis agora.'
    lines = ['   Onde comprar:']
    for link in links:
        price = f' — R${link.price}' if link.price is not None else ''
        lines.append(f'     - {link.store}{price}: {link.url}')
    return '\n'.join(lines)


def _format_caveats(product: Product, budget: float | None) -> str:
    notes = list(product.unmet_requirements)
    price = product.estimated_price
    if budget is not None and price is not None and float(price) > budget:
        notes.insert(0, 'um pouco acima do seu orçamento, mas vale considerar')
    if not notes:
        return ''
    return f'\n   ⚠️ Observação: {"; ".join(notes)}'


def _format_recommendations(products: list[Product], budget: float | None) -> str:
    if not products:
        return (
            'Não consegui encontrar produtos para esses requisitos agora. Você pode '
            'rever ou afrouxar algum requisito (orçamento, características) para eu '
            'tentar de novo?'
        )
    lines = ['Encontrei estas opções com melhor custo-benefício para você:\n']
    for i, product in enumerate(products[:TOP_PRODUCTS], start=1):
        price = (
            f'R${product.estimated_price}'
            if product.estimated_price is not None
            else 'preço não informado'
        )
        features = (
            f'\n   Destaques: {", ".join(product.key_features)}'
            if product.key_features
            else ''
        )
        review = (
            f'\n   Avaliações: {product.review_summary}'
            if product.review_summary
            else ''
        )
        block = (
            f'{i}. {product.name}'
            f'{f" ({product.brand})" if product.brand else ""} — {price}\n'
            f'   Por quê: {product.reason}{features}{review}'
            f'{_format_caveats(product, budget)}\n'
            f'{_format_links(product.purchase_links)}'
        )
        lines.append(block)
    return '\n\n'.join(lines)


# --------------------------------------------------------------------------- #
# Tools do agente de produtos (ReAct). Coleta é conduzida pelo prompt; as
# operações (busca, reviews, apresentação) são estas tools.
# --------------------------------------------------------------------------- #
def build_product_tools() -> list[BaseTool]:
    @tool
    async def search_products(query: str) -> str:
        """Pesquisa produtos na web (Brasil) e retorna candidatos com preço e motivo.

        Use UMA query por chamada e refine a próxima com base no que voltar (modelos
        citados, faixas de preço, alternativas). Inclua o orçamento na query quando
        fizer sentido (ex. "notebook para edição até 4000 reais").

        Args:
            query: Texto de busca em português (produto + necessidade + faixa de preço).

        Returns:
            Uma lista textual de candidatos, um por linha, no formato
            "- Nome (Marca) — R$preço: motivo". Ou um aviso se nada foi encontrado.
        """
        results = await run_planned_searches([query])
        products = await extract_candidates(results, query, None)
        return _format_candidates(products)

    @tool
    async def check_reviews(product_name: str, brand: str | None = None) -> str:
        """Resume a reputação online (avaliações) de um produto.

        Use nos candidatos mais promissores ANTES de finalizar, para embasar a escolha
        e a ordenação, e para poder citar o resumo na recomendação.

        Args:
            product_name: Nome/modelo do produto (ex. "Samsung Galaxy A55").
            brand: Marca/fabricante, se conhecida (ex. "Samsung"). Opcional.

        Returns:
            Uma frase curta com o veredito ("bem avaliado" ou "avaliações fracas ou
            mistas") seguido de um resumo da reputação online.
        """
        product = Product(name=product_name, brand=brand, reason='')
        verdict = await review_verdict(product)
        rating = 'bem avaliado' if verdict.well_rated else 'avaliações fracas ou mistas'
        return f'{rating}: {verdict.summary}'

    @tool
    async def present_recommendations(
        products: list[Product], budget: float | None = None
    ) -> str:
        """Finaliza o atendimento entregando a recomendação ao usuário (ação terminal).

        Chame esta tool UMA vez, ao final, depois de já ter pesquisado e checado
        avaliações. O sistema valida se os anúncios estão no ar, anexa os links de
        compra, aplica a regra de orçamento (quando houver: mantém 4 opções na faixa
        ou mais baratas + 1 opção um pouco acima) e formata o texto final. Garante que
        a resposta nunca fica vazia.

        Args:
            products: Os candidatos escolhidos, do melhor para o pior custo-benefício.
                Preencha por item: `name`, `brand`, `estimated_price` (número em BRL,
                sem símbolo), `reason`, `key_features`, `review_summary` (o que você
                achou em check_reviews) e `unmet_requirements` (requisitos do usuário
                que o produto não atende, se houver). NÃO preencha `available` nem
                `purchase_links` — o sistema preenche.
            budget: Orçamento máximo em reais (BRL) informado pelo usuário, ou None se
                ele não tiver um orçamento definido.

        Returns:
            O texto final formatado com as recomendações, pronto para o usuário.
        """
        enriched = await asyncio.gather(*(_enrich_product(p) for p in products))
        ranked = _rank_by_quality(list(enriched))
        final = select_final_products(ranked, budget)
        return _format_recommendations(final, budget)

    return [search_products, check_reviews, present_recommendations]


# --------------------------------------------------------------------------- #
# Seleção final determinística (regra de orçamento + garantia de não-vazio).
# --------------------------------------------------------------------------- #
def _price_of(product: Product) -> float | None:
    price = product.estimated_price
    return float(price) if price is not None else None


def select_final_products(
    candidates: list[Product], budget: float | None, top: int = TOP_PRODUCTS
) -> list[Product]:
    """Escolhe até `top` produtos aplicando a regra de orçamento.

    - Sem orçamento: os `top` melhores por custo-benefício (ordem do agente).
    - Com orçamento: 1 opção "stretch" (um pouco acima) + as demais na faixa ou baratas.
    - Nunca retorna vazio se houver candidatos: completa com os mais próximos.
    """
    if not candidates or budget is None:
        return candidates[:top]

    # Preço desconhecido conta como elegível à faixa do orçamento.
    within: list[Product] = []
    above: list[tuple[float, Product]] = []
    for product in candidates:
        price = _price_of(product)
        if price is None or price <= budget:
            within.append(product)
        else:
            above.append((price, product))

    tolerance = budget * (1 + BUDGET_STRETCH_TOLERANCE)
    above.sort(key=lambda item: item[0])
    stretch = next((p for price, p in above if price <= tolerance), None)
    if stretch is None and above:
        stretch = above[0][1]

    selected: list[Product] = list(within[: top - 1]) if stretch else list(within[:top])
    if stretch is not None:
        selected.append(stretch)

    # Garantia de não-vazio / completar até `top`: preenche com os que sobraram,
    # priorizando os dentro do orçamento (evita mais de uma opção acima).
    if len(selected) < top:
        chosen = {id(p) for p in selected}
        rest = [*within, *(p for _, p in above)]
        remaining = [p for p in rest if id(p) not in chosen]
        selected.extend(remaining[: top - len(selected)])

    return selected[:top]
