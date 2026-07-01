from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents import build_agent
from app.api import router
from app.infra.db import get_postgres_checkpointer_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Abre o pool/checkpointer e compila o grafo uma única vez, reaproveitando-os
    # entre requisições. O `async with` fecha o pool no shutdown.
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

    app.include_router(router)
    return app
