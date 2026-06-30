from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import Settings


async def get_postgres_checkpointer() -> AsyncPostgresSaver:
    settings = Settings()  # type: ignore
    database_url = settings.CHECKPOINTER_DATABASE_URL

    async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
        await checkpointer.setup()
        return checkpointer
