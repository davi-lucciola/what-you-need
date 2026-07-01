from functools import lru_cache
from typing import TypedDict

from tavily import AsyncTavilyClient

from app.config import Settings


class TavilyImage(TypedDict):
    url: str
    description: str


class TavilyResult(TypedDict):
    title: str
    url: str
    content: str
    score: float
    raw_content: str | None
    favicon: str
    images: list[TavilyImage]


class TavilyAutoParameters(TypedDict):
    topic: str
    search_depth: str


class TavilyUsage(TypedDict):
    credits: int


class TavilySearchResponse(TypedDict):
    query: str
    answer: str
    images: list[TavilyImage]
    results: list[TavilyResult]
    response_time: str
    auto_parameters: TavilyAutoParameters
    usage: TavilyUsage
    request_id: str


class TavilyExtractResult(TypedDict):
    url: str
    raw_content: str | None
    images: list[str]
    favicon: str


class TavilyExtractFailure(TypedDict):
    url: str
    error: str


class TavilyExtractResponse(TypedDict):
    results: list[TavilyExtractResult]
    failed_results: list[TavilyExtractFailure]
    response_time: float


@lru_cache
def get_tavily_client() -> AsyncTavilyClient:
    settings = Settings()  # type: ignore
    return AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
