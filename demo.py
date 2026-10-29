"""
demo.py — Corre los 6 escenarios de prueba de la consigna y genera las trazas.

    python demo.py

Cada escenario usa un thread_id propio (con la fecha y hora de la corrida,
para que dos corridas no se mezclen). Las trazas quedan en traces/ y el
resumen en traces/resumen_demo.json.

El escenario 5 (memoria) se corre en dos partes:
- 5a: dentro de este proceso, dos mensajes con el mismo thread_id.
- 5b y 5c: después de CERRAR la base SQLite, se lanzan procesos NUEVOS de
  main.py (como si cerraras y volvieras a abrir el programa): uno con el mismo
  thread_id (tiene que recordar) y otro con un thread_id distinto (no tiene que
  recordar).
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Any

from src import config
from src.graph import construir_grafo
from src.persistence import abrir_checkpointer
from src.runner import ResultadoTurno, ejecutar_turno

MARCA: str = datetime.now().strftime("%Y%m%d-%H%M%S")
RESUMEN: list[dict[str, Any]] = []


def registrar(escenario: str, thread_id: str, mensaje: str, r: ResultadoTurno) -> None:
    """Muestra el resultado por consola y lo agrega al resumen."""
    herramientas = ", ".join(r.herramientas) or "ninguna"
    print(f"\n{'─' * 78}\n▶ {escenario}  (thread_id={thread_id})\n  Usuario: {mensaje}")
    print(f"  Herramientas: {herramientas} | Estado: {r.estado}")
    print(f"  Respuesta: {r.respuesta[:500]}{'...' if len(r.respuesta) > 500 else ''}")
    print(f"  Traza: {r.traza.relative_to(config.BASE_DIR)}")
    RESUMEN.append({
        "escenario": escenario, "thread_id": thread_id, "mensaje": mensaje,
        "herramientas": r.herramientas, "estado": r.estado,
        "traza": str(r.traza.relative_to(config.BASE_DIR)), "respuesta": r.respuesta,
    })


async def correr_en_proceso_nuevo(escenario: str, thread_id: str, mensaje: str) -> None:
    """Lanza `python main.py --thread-id ... -m ...` como un proceso nuevo."""
    proceso = await asyncio.create_subprocess_exec(
        sys.executable, "main.py", "--thread-id", thread_id, "-m", mensaje, "--escenario", escenario,
        cwd=config.BASE_DIR, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    salida, _ = await proceso.communicate()
    texto = salida.decode("utf-8", errors="replace")
    respuesta = texto.split("Agente: ", 1)[-1].split("\n   [estado:", 1)[0].strip()
    print(f"\n{'─' * 78}\n▶ {escenario}  (PROCESO NUEVO, thread_id={thread_id})\n  Usuario: {mensaje}")
    print(f"  Respuesta: {respuesta[:500]}{'...' if len(respuesta) > 500 else ''}")
    RESUMEN.append({
        "escenario": escenario, "thread_id": thread_id, "mensaje": mensaje,
        "proceso_nuevo": True, "codigo_salida": proceso.returncode, "respuesta": respuesta,
    })


async def main() -> None:
    config.configurar_logs()
    config.validar_entorno()
    hilo_memoria = f"memoria-{MARCA}"

    async with abrir_checkpointer() as checkpointer:
        grafo = construir_grafo(checkpointer)

        casos: list[tuple[str, str, str, int]] = [
            ("1_usa_herramienta", f"esc1-{MARCA}",
             "¿Cómo hago requests asíncronas con httpx?", config.RECURSION_LIMIT),
            ("2_sin_herramienta", f"esc2-{MARCA}",
             "¡Hola! ¿Quién sos y en qué me podés ayudar?", config.RECURSION_LIMIT),
            ("3_multi_paso", f"esc3-{MARCA}",
             "Buscá cómo se habilita HTTP/2 en httpx y después mostrame el texto completo "
             "del fragmento más relevante.", config.RECURSION_LIMIT),
            ("4a_reintento_id_invalido", f"esc4a-{MARCA}",
             "Mostrame el fragmento async-p7-c0 de la documentación.", config.RECURSION_LIMIT),
            ("4b_info_incompleta", f"esc4b-{MARCA}",
             "¿Cómo abro una conexión WebSocket con httpx?", config.RECURSION_LIMIT),
            ("5a_memoria_pregunta_1", hilo_memoria,
             "¿Qué es un AsyncClient en httpx?", config.RECURSION_LIMIT),
            ("5a_memoria_pregunta_2", hilo_memoria,
             "¿Y cómo se usa?", config.RECURSION_LIMIT),
            ("6_recursion_limit", f"esc6-{MARCA}",
             "Comparame en detalle cómo se configuran en httpx los proxies, la autenticación, "
             "los timeouts y HTTP/2.", 3),
        ]
        for escenario, thread_id, mensaje, limite in casos:
            r = await ejecutar_turno(grafo, thread_id, mensaje, escenario, recursion_limit=limite)
            registrar(escenario, thread_id, mensaje, r)
    # Acá la base SQLite ya está CERRADA: lo que sigue corre en procesos nuevos.

    await correr_en_proceso_nuevo(
        "5b_memoria_tras_reiniciar_mismo_thread", hilo_memoria,
        "Recordame: ¿de qué clase de httpx estábamos hablando y qué te pregunté recién?",
    )
    await correr_en_proceso_nuevo(
        "5c_memoria_otro_thread", f"otro-{MARCA}",
        "Recordame: ¿de qué clase de httpx estábamos hablando y qué te pregunté recién?",
    )

    resumen = config.TRACES_DIR / "resumen_demo.json"
    resumen.write_text(json.dumps({"corrida": MARCA, "escenarios": RESUMEN}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\n{'═' * 78}\nListo. Resumen en {resumen.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except config.ErrorDeConfiguracion as e:
        print(f"\n❌ ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)
