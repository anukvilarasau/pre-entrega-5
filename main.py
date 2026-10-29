"""
main.py — Chat por consola con el agente técnico de httpx.

Uso:
    python main.py --thread-id mi-sesion            # modo interactivo
    python main.py --thread-id mi-sesion -m "hola"  # un solo mensaje y sale

Con el mismo --thread-id el agente recuerda la conversación, aunque cierres
y vuelvas a abrir el programa (la memoria vive en data/checkpoints.sqlite).
"""

import argparse
import asyncio
import logging
import sys

from src import config
from src.graph import construir_grafo
from src.persistence import abrir_checkpointer
from src.runner import ResultadoTurno, ejecutar_turno, explicar_error

log = logging.getLogger("main")
COMANDOS_SALIR: set[str] = {"salir", "exit", "quit"}


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asistente técnico de httpx (LangGraph + SqliteSaver).")
    parser.add_argument("--thread-id", default="sesion-principal",
                        help="ID de la conversación (misma ID = misma memoria).")
    parser.add_argument("-m", "--mensaje", default=None,
                        help="Manda un solo mensaje, muestra la respuesta y termina.")
    parser.add_argument("--recursion-limit", type=int, default=config.RECURSION_LIMIT,
                        help=f"Techo de pasos del grafo por mensaje (por defecto {config.RECURSION_LIMIT}).")
    parser.add_argument("--escenario", default="",
                        help="Etiqueta para el nombre del archivo de traza (la usa demo.py).")
    return parser.parse_args()


def mostrar(resultado: ResultadoTurno) -> None:
    """Imprime la respuesta y un resumen de lo que hizo el agente."""
    herramientas = ", ".join(resultado.herramientas) or "ninguna"
    print(f"\nAgente: {resultado.respuesta}\n")
    print(f"   [estado: {resultado.estado} | herramientas: {herramientas} | "
          f"traza: {resultado.traza.relative_to(config.BASE_DIR)}]\n")


async def main() -> int:
    args = leer_argumentos()
    config.configurar_logs()
    try:
        config.validar_entorno()
    except config.ErrorDeConfiguracion as e:
        print(f"\n❌ ERROR: {e}\n", file=sys.stderr)
        return 1

    async with abrir_checkpointer() as checkpointer:
        grafo = construir_grafo(checkpointer)

        if args.mensaje:  # modo de un solo mensaje
            print(f"\nVos ({args.thread_id}): {args.mensaje}")
            try:
                mostrar(await ejecutar_turno(grafo, args.thread_id, args.mensaje, args.escenario,
                                             recursion_limit=args.recursion_limit))
            except Exception as e:
                print(f"\n❌ ERROR: {explicar_error(e)}\n", file=sys.stderr)
                return 1
            return 0

        print(f"\nAsistente técnico de httpx — thread_id: '{args.thread_id}'. Escribí 'salir' para terminar.\n")
        while True:
            try:
                texto = (await asyncio.to_thread(input, "Vos: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not texto:
                continue
            if texto.lower() in COMANDOS_SALIR:
                break
            try:
                mostrar(await ejecutar_turno(grafo, args.thread_id, texto,
                                             recursion_limit=args.recursion_limit))
            except Exception as e:  # la API no responde, clave inválida, etc.: el chat sigue vivo
                print(f"\n❌ ERROR: {explicar_error(e)}\n   Probá de nuevo.\n", file=sys.stderr)

    print("Chau. La conversación quedó guardada; retomala con el mismo --thread-id.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
