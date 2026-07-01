from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import Settings


@asynccontextmanager
async def get_postgres_checkpointer_pool() -> AsyncGenerator[AsyncPostgresSaver]:
    """Checkpointer com pool de conexões, para servidores web concorrentes.

    Um `AsyncConnectionPool` atende várias requisições HTTP em paralelo. As
    `kwargs` (`autocommit` + `prepare_threshold=0`) e o `row_factory=dict_row`
    são requisitos do `AsyncPostgresSaver`.
    """
    settings = Settings()  # type: ignore
    database_url = settings.CHECKPOINTER_DATABASE_URL
    kwargs = {
        'autocommit': True,
        'prepare_threshold': 0,
        'row_factory': dict_row,
    }

    async with AsyncConnectionPool(database_url, kwargs=kwargs) as pool:
        checkpointer = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await checkpointer.setup()
        yield checkpointer
