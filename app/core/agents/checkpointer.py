import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import Settings

logger = logging.getLogger(__name__)


async def _ensure_database_reachable(database_url: str) -> None:
    """Fail-fast: testa a conexão antes de abrir o pool.

    Uma única tentativa direta (com `connect_timeout` curto) produz um erro
    limpo, sem os workers do pool spammando `error connecting in 'pool-1'` nem o
    traceback longo de `PoolTimeout`. Em caso de falha, encerra o processo com
    uma mensagem amigável — usando `os._exit` para não deixar o uvicorn logar o
    traceback do lifespan (o pool ainda não foi aberto, então não há cleanup).
    """
    conn = await psycopg.AsyncConnection.connect(database_url, connect_timeout=5)
    await conn.close()


@asynccontextmanager
async def get_postgres_checkpointer_pool() -> AsyncGenerator[AsyncPostgresSaver]:
    """Checkpointer com pool de conexões, para servidores web concorrentes.

    Um `AsyncConnectionPool` atende várias requisições HTTP em paralelo. As
    `kwargs` (`autocommit` + `prepare_threshold=0`) e o `row_factory=dict_row`
    são requisitos do `AsyncPostgresSaver`.
    """
    settings = Settings()  # type: ignore
    database_url = settings.CHECKPOINTER_DATABASE_URL
    kwargs = {'autocommit': True, 'prepare_threshold': 0, 'row_factory': dict_row}

    await _ensure_database_reachable(database_url)

    async with AsyncConnectionPool(database_url, kwargs=kwargs) as pool:
        checkpointer = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await checkpointer.setup()
        yield checkpointer
