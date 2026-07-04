from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat_router
from app.core.agents import build_agent
from app.infra.checkpointer import get_postgres_checkpointer_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    async with get_postgres_checkpointer_pool() as checkpointer:
        app.state.agent = build_agent(checkpointer=checkpointer)
        yield


def create_app() -> FastAPI:
    app = FastAPI(title='Assistente de busca de produtos', lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    app.include_router(chat_router)
    return app
