from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import Settings


@asynccontextmanager
async def get_postgres_checkpointer() -> AsyncGenerator[AsyncPostgresSaver]:
    settings = Settings()  # type: ignore
    database_url = settings.CHECKPOINTER_DATABASE_URL

    async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
