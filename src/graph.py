"""
graph.py — El grafo ReAct del agente (LangGraph StateGraph).

    START → agent ──(¿pidió herramientas?)──► tools ──► agent ... → END

- Nodo "agent": el LLM (Claude) con las herramientas "enchufadas" (bind_tools).
  ÉL decide si responde directo o si llama a una herramienta. No hay ningún
  if/else nuestro que elija la ruta según la pregunta.
- Nodo "tools": ToolNode ejecuta las herramientas que pidió el LLM y devuelve
  los resultados (observaciones) como ToolMessage.
- tools_condition: si el último mensaje del LLM tiene tool_calls va a "tools";
  si no, termina (END). Por eso el ciclo corta solo cuando el LLM ya no pide
  más herramientas.
"""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr

from src import config
from src.state import AgentState
from src.tools import HERRAMIENTAS

log = logging.getLogger(__name__)

PROMPT_SISTEMA: str = """Sos un asistente técnico que responde preguntas sobre la librería httpx de Python, \
usando su documentación oficial (en inglés), indexada en una base vectorial.

Herramientas:
- buscar_documentacion: busca fragmentos relevantes. Las consultas van SIEMPRE en inglés.
- obtener_fragmento: lee el texto completo de un fragmento por su ID exacto.

Cómo trabajar:
1. Decidí vos si necesitás una herramienta. Para saludos, agradecimientos o charla general \
respondé directamente, sin herramientas. Para cualquier pregunta técnica sobre httpx, buscá en \
la documentación: no respondas de memoria.
2. Antes de cada llamada a una herramienta, escribí UNA línea corta con lo que vas a hacer y por qué \
(tu razonamiento). Después de recibir el resultado, decidí si alcanza o si necesitás otro paso.
3. Si una herramienta devuelve "INFORMACIÓN INCOMPLETA" o "ERROR", no te rindas al primer intento: \
reintentá UNA vez con otros argumentos (otras palabras en inglés, un ID sugerido, etc.). Si el \
segundo intento tampoco sirve, decí claramente que no está en la documentación y pedile al usuario \
que aclare o reformule la pregunta.
4. Si la pregunta tiene varias partes, hacé las búsquedas que hagan falta antes de responder.
5. Usá el historial de la conversación: si el usuario dice "¿y cómo se usa?", se refiere al tema anterior.
6. Respondé en español rioplatense, breve y claro, con ejemplos de código cuando sirvan, y citá \
los IDs de los fragmentos que usaste entre corchetes, por ejemplo [timeouts-p1-c0]."""

AVISO_CANCELADA: str = "Llamada cancelada: el agente alcanzó el límite de pasos (recursion_limit) antes de ejecutarla."


def crear_llm() -> Runnable[Any, BaseMessage]:
    """Crea Claude con las herramientas enchufadas.

    Nota: la consigna pide temperatura 0, pero desde octubre de 2026 la API de
    Claude rechaza el parámetro `temperature` en los modelos actuales. Se usa
    effort="low" (respuestas directas y estables) como reemplazo.
    """
    llm: BaseChatModel = ChatAnthropic(
        model_name=config.ANTHROPIC_MODEL,
        api_key=SecretStr(config.ANTHROPIC_API_KEY),
        max_tokens_to_sample=config.MAX_TOKENS_RESPUESTA,
        effort="low",
        max_retries=2,
        timeout=60,
        stop=None,
    )
    return llm.bind_tools(HERRAMIENTAS)


def reparar_llamadas_huerfanas(mensajes: list[BaseMessage]) -> list[BaseMessage]:
    """Agrega un ToolMessage de "cancelada" a cada tool_call que quedó sin respuesta.

    Pasa si una corrida se cortó por recursion_limit justo después de que el
    LLM pidió una herramienta. La API de Claude exige que cada pedido de
    herramienta tenga su resultado, así que sin esto el hilo quedaría "roto".
    """
    respondidas: set[str] = {m.tool_call_id for m in mensajes if isinstance(m, ToolMessage)}
    reparados: list[BaseMessage] = []
    for m in mensajes:
        reparados.append(m)
        if isinstance(m, AIMessage):
            for llamada in m.tool_calls:
                if llamada["id"] not in respondidas:
                    reparados.append(ToolMessage(content=AVISO_CANCELADA, tool_call_id=llamada["id"]))
    return reparados


def preparar_historial(mensajes: list[BaseMessage]) -> list[BaseMessage]:
    """Limpia y recorta el historial ANTES de mandarlo al modelo (evita "estado sucio").

    - Repara pedidos de herramientas sin respuesta.
    - Si el historial supera MAX_TOKENS_HISTORIAL, se queda con los mensajes
      más recientes, empezando siempre en un mensaje del usuario (así nunca
      queda un resultado de herramienta sin su pedido).
    El estado guardado en SQLite NO se modifica: solo lo que ve el modelo.
    """
    limpios = reparar_llamadas_huerfanas(mensajes)
    recortados: list[BaseMessage] = trim_messages(
        limpios,
        max_tokens=config.MAX_TOKENS_HISTORIAL,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    if len(recortados) < len(limpios):
        log.info("Historial recortado: %d → %d mensajes", len(limpios), len(recortados))
    return recortados


def construir_grafo(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    llm: Runnable[Any, BaseMessage] | None = None,
) -> CompiledStateGraph:
    """Arma y compila el StateGraph. Con checkpointer, el estado se guarda por thread_id."""
    modelo = llm if llm is not None else crear_llm()

    async def agente(state: AgentState) -> dict[str, list[BaseMessage]]:
        """Nodo 'agent': el LLM lee el historial y decide (responder o usar herramientas)."""
        historial = preparar_historial(state["messages"])
        respuesta = await modelo.ainvoke([SystemMessage(content=PROMPT_SISTEMA), *historial])
        return {"messages": [respuesta]}

    grafo = StateGraph(AgentState)
    grafo.add_node("agent", agente)
    grafo.add_node("tools", ToolNode(HERRAMIENTAS))

    grafo.add_edge(START, "agent")
    # tools_condition: "tools" si el LLM pidió herramientas; END si no.
    grafo.add_conditional_edges("agent", tools_condition)
    grafo.add_edge("tools", "agent")  # después de cada herramienta, vuelve a pensar

    return grafo.compile(checkpointer=checkpointer)
