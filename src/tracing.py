"""
tracing.py — Guarda la traza ReAct de cada corrida en traces/.

ReAct = Razonar + Actuar: el agente piensa (Thought), llama a una herramienta
(Action), lee el resultado (Observation) y repite hasta dar la respuesta final.
Cada corrida queda en dos archivos con el mismo nombre:
- .json: estructurado (pasos, argumentos, observaciones, tokens, timestamps);
- .log: el mismo recorrido en texto legible.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from src import config


def ahora() -> str:
    """Timestamp ISO con milisegundos y zona horaria local."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


@dataclass
class Paso:
    """Un paso de la traza: un pensamiento/acción del LLM o una observación."""

    n: int
    tipo: str  # "accion" | "observacion" | "respuesta_final" | "corte_recursion_limit"
    timestamp: str
    pensamiento: str | None = None
    herramienta: str | None = None
    argumentos: dict[str, Any] | None = None
    tool_call_id: str | None = None
    observacion: str | None = None
    tokens: dict[str, int] | None = None


@dataclass
class Traza:
    """Traza completa de una corrida (un mensaje del usuario y todo lo que siguió)."""

    thread_id: str
    pregunta: str
    escenario: str = ""
    modelo: str = config.ANTHROPIC_MODEL
    recursion_limit: int = config.RECURSION_LIMIT
    inicio: str = field(default_factory=ahora)
    fin: str = ""
    duracion_s: float = 0.0
    estado: str = "en_curso"  # "completo" | "cortado_por_recursion_limit" | "error"
    pasos: list[Paso] = field(default_factory=list)
    respuesta_final: str = ""
    herramientas_usadas: list[str] = field(default_factory=list)
    tokens_totales: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )

    # ------------------------------------------------------------------
    # Registro de pasos
    # ------------------------------------------------------------------
    def registrar_ai(self, mensaje: AIMessage) -> None:
        """Registra un mensaje del LLM: pensamiento + llamadas, o la respuesta final."""
        texto = mensaje.text.strip()
        tokens = self._sumar_tokens(mensaje)
        if mensaje.tool_calls:
            for i, llamada in enumerate(mensaje.tool_calls):
                self.pasos.append(Paso(
                    n=len(self.pasos) + 1,
                    tipo="accion",
                    timestamp=ahora(),
                    pensamiento=texto if i == 0 and texto else None,
                    herramienta=llamada["name"],
                    argumentos=dict(llamada["args"]),
                    tool_call_id=llamada["id"],
                    tokens=tokens if i == 0 else None,
                ))
        else:
            self.pasos.append(Paso(
                n=len(self.pasos) + 1, tipo="respuesta_final", timestamp=ahora(), tokens=tokens,
            ))
            self.respuesta_final = texto

    def registrar_observacion(self, mensaje: ToolMessage) -> None:
        """Registra el resultado de una herramienta (lo que "observa" el agente)."""
        self.pasos.append(Paso(
            n=len(self.pasos) + 1,
            tipo="observacion",
            timestamp=ahora(),
            herramienta=mensaje.name,
            tool_call_id=mensaje.tool_call_id,
            observacion=str(mensaje.content),
        ))
        self.herramientas_usadas.append(mensaje.name or "?")  # solo las que se ejecutaron

    def registrar_corte(self, aviso: str) -> None:
        """Registra el corte por recursion_limit (las acciones pendientes no se ejecutaron)."""
        self.pasos.append(Paso(
            n=len(self.pasos) + 1, tipo="corte_recursion_limit", timestamp=ahora(), observacion=aviso,
        ))

    def _sumar_tokens(self, mensaje: AIMessage) -> dict[str, int] | None:
        uso = mensaje.usage_metadata
        if not uso:
            return None
        tokens = {
            "input_tokens": int(uso.get("input_tokens", 0)),
            "output_tokens": int(uso.get("output_tokens", 0)),
            "total_tokens": int(uso.get("total_tokens", 0)),
        }
        for clave, valor in tokens.items():
            self.tokens_totales[clave] += valor
        return tokens

    def cerrar(self, estado: str, respuesta: str | None = None) -> None:
        """Marca el final de la corrida."""
        self.estado = estado
        if respuesta is not None:
            self.respuesta_final = respuesta
        self.fin = ahora()
        self.duracion_s = round(
            (datetime.fromisoformat(self.fin) - datetime.fromisoformat(self.inicio)).total_seconds(), 2
        )

    # ------------------------------------------------------------------
    # Guardado
    # ------------------------------------------------------------------
    def guardar(self, carpeta: Path = config.TRACES_DIR) -> Path:
        """Escribe <nombre>.json y <nombre>.log en traces/ y devuelve la ruta del .json."""
        carpeta.mkdir(parents=True, exist_ok=True)
        marca = datetime.fromisoformat(self.inicio).strftime("%Y%m%d-%H%M%S")
        etiqueta = re.sub(r"[^\w-]+", "_", self.escenario or self.thread_id).strip("_")
        base = carpeta / f"{etiqueta}_{marca}"
        base.with_suffix(".json").write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        base.with_suffix(".log").write_text(self.como_texto(), encoding="utf-8")
        return base.with_suffix(".json")

    def como_texto(self) -> str:
        """Versión legible de la traza (formato Thought / Action / Observation)."""
        lineas: list[str] = [
            "=" * 78,
            f"TRAZA ReAct | escenario: {self.escenario or '-'} | thread_id: {self.thread_id}",
            f"modelo: {self.modelo} | recursion_limit: {self.recursion_limit} | inicio: {self.inicio}",
            "=" * 78,
            f"USUARIO: {self.pregunta}",
            "",
        ]
        for p in self.pasos:
            hora = p.timestamp[11:23]
            if p.tipo == "accion":
                if p.pensamiento:
                    lineas.append(f"[{hora}] PENSAMIENTO: {p.pensamiento}")
                args = json.dumps(p.argumentos, ensure_ascii=False)
                lineas.append(f"[{hora}] ACCIÓN: {p.herramienta}({args})")
            elif p.tipo == "observacion":
                obs = (p.observacion or "").replace("\n", " ")
                lineas.append(f"[{hora}] OBSERVACIÓN ({p.herramienta}): {obs[:400]}{'...' if len(obs) > 400 else ''}")
            elif p.tipo == "corte_recursion_limit":
                lineas.append(f"[{hora}] CORTE POR recursion_limit: las acciones pendientes NO se ejecutaron")
            else:
                lineas.append(f"[{hora}] RESPUESTA FINAL (sin más herramientas)")
            if p.tokens:
                lineas.append(f"           tokens: in={p.tokens['input_tokens']} out={p.tokens['output_tokens']}")
        lineas += [
            "",
            f"RESPUESTA: {self.respuesta_final}",
            "",
            f"ESTADO: {self.estado} | DURACIÓN: {self.duracion_s} s | HERRAMIENTAS EJECUTADAS: "
            f"{', '.join(self.herramientas_usadas) or 'ninguna'}",
            f"TOKENS TOTALES: in={self.tokens_totales['input_tokens']} "
            f"out={self.tokens_totales['output_tokens']} total={self.tokens_totales['total_tokens']}",
            "",
        ]
        return "\n".join(lineas)
