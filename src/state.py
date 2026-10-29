"""
state.py — El estado del agente: la "ficha" que viaja por el grafo.

Solo contiene DATOS (nada de clientes, conexiones ni modelos), porque el
estado se guarda tal cual en SQLite después de cada paso. Si tuviera un
cliente de Pinecone adentro, no se podría serializar.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Estado del agente.

    messages: historial completo de la conversación (usuario, modelo y
    resultados de herramientas). `add_messages` es el "reductor": cuando un
    nodo devuelve mensajes nuevos, se AGREGAN al final en lugar de pisar la
    lista. Así cada vuelta del ciclo suma su paso al historial.
    """

    messages: Annotated[list[BaseMessage], add_messages]
