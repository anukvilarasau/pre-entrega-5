"""
runner.py — Ejecuta UN turno de conversación con el agente y guarda su traza.

Lo usan main.py (modo interactivo) y demo.py (escenarios de prueba), así la
forma de invocar el grafo (thread_id + recursion_limit) y el manejo de
errores están escritos una sola vez.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src import config
from src.graph import AVISO_CANCELADA
from src.persistence import config_de_hilo
from src.tracing import Traza

log = logging.getLogger(__name__)


@dataclass
class ResultadoTurno:
    """Lo que devuelve un turno: la respuesta, cómo terminó y dónde quedó la traza."""

    respuesta: str
    estado: str
    traza: Path
    herramientas: list[str]


async def ejecutar_turno(
    grafo: CompiledStateGraph,
    thread_id: str,
    mensaje: str,
    escenario: str = "",
    recursion_limit: int = config.RECURSION_LIMIT,
) -> ResultadoTurno:
    """Manda un mensaje al agente en el hilo `thread_id` y registra cada paso ReAct.

    Se usa astream(stream_mode="updates"): cada vez que un nodo termina, llega
    lo que agregó al estado. Así se registra cada pensamiento, acción y
    observación con su propio timestamp.
    """
    traza = Traza(thread_id=thread_id, pregunta=mensaje, escenario=escenario,
                  recursion_limit=recursion_limit)
    cfg: RunnableConfig = config_de_hilo(thread_id, recursion_limit)
    try:
        async for actualizacion in grafo.astream(
            {"messages": [HumanMessage(content=mensaje)]}, cfg, stream_mode="updates"
        ):
            for nodo, salida in actualizacion.items():
                for m in (salida or {}).get("messages", []):
                    if isinstance(m, AIMessage):
                        traza.registrar_ai(m)
                    elif isinstance(m, ToolMessage):
                        traza.registrar_observacion(m)
                log.debug("nodo %s terminó", nodo)
        traza.cerrar("completo")
    except GraphRecursionError:
        aviso = (
            f"Me detuve de forma controlada: alcancé el límite de {recursion_limit} pasos "
            "(recursion_limit) antes de terminar. Probá con una pregunta más concreta."
        )
        await cerrar_hilo_cortado(grafo, cfg, aviso)
        traza.registrar_corte(aviso)
        traza.cerrar("cortado_por_recursion_limit", aviso)
    except Exception as e:  # la API de Claude no responde, clave inválida, etc.
        log.error("El turno falló: %s: %s", type(e).__name__, e)
        traza.cerrar("error", explicar_error(e))
        raise
    finally:
        ruta = traza.guardar()

    return ResultadoTurno(traza.respuesta_final, traza.estado, ruta, traza.herramientas_usadas)


def explicar_error(e: Exception) -> str:
    """Traduce los errores más comunes de la API de Claude a un mensaje claro."""
    if isinstance(e, anthropic.AuthenticationError):
        return "La ANTHROPIC_API_KEY del .env es inválida (la API respondió 401)."
    if isinstance(e, anthropic.NotFoundError):
        return f"El modelo '{config.ANTHROPIC_MODEL}' no existe o no tenés acceso. Revisá ANTHROPIC_MODEL en el .env."
    if isinstance(e, anthropic.RateLimitError):
        return "La API de Claude está limitando los pedidos (429). Esperá un momento y probá de nuevo."
    if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return "No se pudo conectar con la API de Claude (sin internet o timeout)."
    return f"{type(e).__name__}: {e}"


async def cerrar_hilo_cortado(grafo: CompiledStateGraph, cfg: RunnableConfig, aviso: str) -> None:
    """Deja el hilo "limpio" después de un corte por recursion_limit.

    Si el corte ocurrió justo después de que el LLM pidió una herramienta, ese
    pedido queda sin respuesta. Se agregan los resultados "cancelada" y el
    aviso como mensaje del agente, para que el próximo mensaje del usuario en
    el mismo thread_id funcione normalmente.
    """
    estado = await grafo.aget_state(cfg)
    mensajes = estado.values.get("messages", [])
    respondidas = {m.tool_call_id for m in mensajes if isinstance(m, ToolMessage)}
    pendientes = [
        ToolMessage(content=AVISO_CANCELADA, tool_call_id=llamada["id"], name=llamada["name"])
        for m in mensajes if isinstance(m, AIMessage)
        for llamada in m.tool_calls if llamada["id"] not in respondidas
    ]
    await grafo.aupdate_state(cfg, {"messages": [*pendientes, AIMessage(content=aviso)]}, as_node="agent")
