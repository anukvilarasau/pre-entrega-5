"""
persistence.py — Memoria persistente del agente con SqliteSaver.

AsyncSqliteSaver (paquete langgraph-checkpoint-sqlite) guarda una "foto" del
estado después de CADA paso del grafo en data/checkpoints.sqlite, separada
por thread_id. Al volver a correr el script con el mismo thread_id, el grafo
recupera la última foto y la conversación sigue donde había quedado.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src import config


@asynccontextmanager
async def abrir_checkpointer(ruta: Path = config.CHECKPOINTS_DB) -> AsyncIterator[AsyncSqliteSaver]:
    """Abre (o crea) la base SQLite de checkpoints y la cierra al terminar.

    Uso:
        async with abrir_checkpointer() as checkpointer:
            grafo = construir_grafo(checkpointer)
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(ruta)) as checkpointer:
        await checkpointer.setup()  # crea las tablas si no existen
        yield checkpointer


def config_de_hilo(thread_id: str, recursion_limit: int = config.RECURSION_LIMIT) -> RunnableConfig:
    """Config de cada invocación: thread_id (qué memoria usar) y techo de pasos."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
