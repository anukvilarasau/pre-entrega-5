"""
tools.py — Las herramientas que el agente puede decidir usar.

Cada herramienta tiene:
- un esquema de entrada en Pydantic (args_schema): el LLM ve qué argumentos
  existen, qué significan y qué límites tienen;
- un docstring MUY descriptivo: es lo que el LLM lee para decidir CUÁNDO usarla;
- try/except: los errores vuelven como TEXTO al LLM (para que razone y
  reintente); nunca rompen el programa.

Los clientes (Pinecone y el modelo de embeddings) NO van en el estado del
grafo: se crean una sola vez, a pedido, y quedan guardados en este módulo.
"""

import asyncio
import logging
import re
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src import config

log = logging.getLogger(__name__)

# Formato de los IDs de fragmento creados en la Pre-entrega 4: <doc>-p<página>-c<n>
PATRON_CHUNK_ID: re.Pattern[str] = re.compile(r"^[a-z0-9_]+-p\d+-c\d+$")
LARGO_EXTRACTO: int = 700  # caracteres de cada resultado de búsqueda


# ---------------------------------------------------------------------------
# Recursos compartidos (se cargan una vez, la primera vez que se usan)
# ---------------------------------------------------------------------------
class _Recursos:
    """Guarda el índice de Pinecone y el modelo de embeddings ya cargados."""

    indice: Any = None
    embeddings: Any = None
    lock: asyncio.Lock | None = None


def _cargar_recursos_sync() -> tuple[Any, Any]:
    """Conecta con Pinecone y carga el modelo de embeddings (código bloqueante)."""
    from huggingface_hub.utils import logging as hf_logging
    from langchain_huggingface import HuggingFaceEmbeddings
    from pinecone import Pinecone

    hf_logging.set_verbosity_error()
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if not pc.has_index(config.INDEX_NAME):
        raise RuntimeError(f"El índice '{config.INDEX_NAME}' no existe en Pinecone.")
    indice = pc.Index(config.INDEX_NAME)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    return indice, embeddings


async def obtener_recursos() -> tuple[Any, Any]:
    """Devuelve (índice, embeddings), cargándolos la primera vez.

    La carga es bloqueante (red + modelo en disco), así que corre en un hilo
    aparte con asyncio.to_thread para no frenar el event loop.
    """
    if _Recursos.lock is None:
        _Recursos.lock = asyncio.Lock()
    async with _Recursos.lock:
        if _Recursos.indice is None:
            log.info("Conectando con Pinecone y cargando embeddings...")
            _Recursos.indice, _Recursos.embeddings = await asyncio.to_thread(_cargar_recursos_sync)
    return _Recursos.indice, _Recursos.embeddings


# ---------------------------------------------------------------------------
# Herramienta 1: búsqueda semántica en la documentación (RAG)
# ---------------------------------------------------------------------------
class BuscarDocumentacionInput(BaseModel):
    """Argumentos de buscar_documentacion."""

    query: str = Field(
        description=(
            "Consulta de búsqueda EN INGLÉS (la documentación está en inglés). "
            "Usá palabras técnicas concretas, por ejemplo 'set a timeout on a request' "
            "o 'AsyncClient usage'. Entre 3 y 200 caracteres."
        ),
        min_length=3,
        max_length=200,
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Cantidad máxima de fragmentos a devolver (entre 1 y 8). Por defecto 3.",
    )


@tool("buscar_documentacion", args_schema=BuscarDocumentacionInput)
async def buscar_documentacion(query: str, limit: int = 3) -> str:
    """Busca en la documentación oficial de la librería httpx (Python) indexada en Pinecone.

    Cuándo usarla: siempre que el usuario pregunte algo técnico sobre httpx
    (clientes, async, timeouts, proxies, autenticación, HTTP/2, streaming,
    excepciones, etc.). NO la uses para saludos, charla o temas que no sean httpx.

    Qué devuelve: hasta `limit` fragmentos relevantes, cada uno con su ID
    (por ejemplo 'async-p1-c0'), su similitud (0 a 1) y un extracto del texto.
    Si ningún fragmento supera el umbral de relevancia devuelve un mensaje que
    empieza con "INFORMACIÓN INCOMPLETA": en ese caso reformulá la consulta en
    inglés con otras palabras (sinónimos, nombre de la clase o del parámetro) y
    volvé a buscar; si después de un segundo intento sigue sin resultados,
    pedile al usuario que aclare su pregunta.
    Para leer un fragmento completo, usá obtener_fragmento con su ID.
    """
    try:
        indice, embeddings = await obtener_recursos()
        vector: list[float] = await asyncio.to_thread(embeddings.embed_query, query)
        respuesta = await asyncio.to_thread(
            indice.query,
            vector=vector,
            top_k=limit,
            namespace=config.PINECONE_NAMESPACE,
            include_metadata=True,
        )
        coincidencias = respuesta.matches or []
        relevantes = [m for m in coincidencias if (m.score or 0.0) >= config.UMBRAL_RELEVANCIA]

        if not relevantes:
            mejor = max((m.score or 0.0 for m in coincidencias), default=0.0)
            return (
                f"INFORMACIÓN INCOMPLETA: ningún fragmento es relevante para '{query}' "
                f"(mejor similitud {mejor:.2f}, umbral {config.UMBRAL_RELEVANCIA:.2f}). "
                "Probá reformular la consulta en inglés con otros términos técnicos, "
                "o pedile al usuario que aclare qué necesita."
            )

        bloques: list[str] = [f"{len(relevantes)} fragmento(s) relevante(s) para '{query}':"]
        for m in relevantes:
            meta: dict[str, Any] = dict(m.metadata or {})
            texto: str = str(meta.get("text", "")).strip()
            extracto = texto[:LARGO_EXTRACTO] + ("..." if len(texto) > LARGO_EXTRACTO else "")
            bloques.append(
                f"\n[ID: {m.id}] similitud={m.score:.2f} | fuente={meta.get('source', '?')}\n{extracto}"
            )
        return "\n".join(bloques)
    except Exception as e:  # Pinecone caído, sin internet, clave inválida, etc.
        log.warning("buscar_documentacion falló: %s", e)
        return (
            f"ERROR al buscar en la documentación: {type(e).__name__}: {e}. "
            "El servicio de búsqueda (Pinecone) no respondió. Avisale al usuario."
        )


# ---------------------------------------------------------------------------
# Herramienta 2: leer un fragmento completo por su ID (determinística)
# ---------------------------------------------------------------------------
class ObtenerFragmentoInput(BaseModel):
    """Argumentos de obtener_fragmento."""

    chunk_id: str = Field(
        description=(
            "ID exacto del fragmento, con el formato '<documento>-p<página>-c<número>', "
            "por ejemplo 'async-p1-c0' o 'timeouts-p1-c1'. Se obtiene de los resultados "
            "de buscar_documentacion o lo da el usuario."
        ),
        min_length=1,
        max_length=60,
    )


@tool("obtener_fragmento", args_schema=ObtenerFragmentoInput)
async def obtener_fragmento(chunk_id: str) -> str:
    """Devuelve el texto COMPLETO de un fragmento de la documentación de httpx a partir de su ID exacto.

    Cuándo usarla: cuando el usuario pide un fragmento por su ID, o cuando un
    extracto de buscar_documentacion quedó cortado ("...") y necesitás el texto
    entero para responder bien (por ejemplo, para mostrar un ejemplo de código
    completo).

    Es determinística: el mismo ID siempre devuelve el mismo texto.
    Qué devuelve: el ID, la fuente, la categoría y el texto completo.
    Si el ID tiene un formato inválido o no existe, devuelve un mensaje que
    empieza con "ERROR" y, si puede, sugiere IDs válidos del mismo documento:
    en ese caso reintentá con uno de esos IDs o usá buscar_documentacion.
    """
    try:
        chunk_id = chunk_id.strip().lower()
        if not PATRON_CHUNK_ID.match(chunk_id):
            return (
                f"ERROR: '{chunk_id}' no tiene el formato de un ID de fragmento. "
                "El formato es '<documento>-p<página>-c<número>', por ejemplo 'async-p1-c0'. "
                "Si no conocés el ID, usá buscar_documentacion primero."
            )

        indice, _ = await obtener_recursos()
        respuesta = await asyncio.to_thread(
            indice.fetch, ids=[chunk_id], namespace=config.PINECONE_NAMESPACE
        )
        vector = respuesta.vectors.get(chunk_id)
        if vector is not None:
            meta: dict[str, Any] = dict(vector.metadata or {})
            return (
                f"[ID: {chunk_id}] fuente={meta.get('source', '?')} | "
                f"categoría={meta.get('category', '?')}\n{str(meta.get('text', '')).strip()}"
            )

        # No existe: buscamos los IDs válidos del mismo documento para sugerirlos.
        documento = chunk_id.split("-p")[0]
        sugeridos: list[str] = []
        for pagina in await asyncio.to_thread(
            lambda: list(indice.list(prefix=f"{documento}-", namespace=config.PINECONE_NAMESPACE))
        ):
            sugeridos.extend(pagina)
        if sugeridos:
            return (
                f"ERROR: el fragmento '{chunk_id}' no existe. IDs válidos del documento "
                f"'{documento}': {', '.join(sorted(sugeridos))}."
            )
        return (
            f"ERROR: el fragmento '{chunk_id}' no existe y no hay ningún documento llamado "
            f"'{documento}'. Usá buscar_documentacion para encontrar IDs válidos."
        )
    except Exception as e:  # Pinecone caído, sin internet, etc.
        log.warning("obtener_fragmento falló: %s", e)
        return f"ERROR al leer el fragmento: {type(e).__name__}: {e}. Avisale al usuario."


# Lista de herramientas que se le "enchufan" al LLM y al ToolNode.
HERRAMIENTAS: list[BaseTool] = [buscar_documentacion, obtener_fragmento]
