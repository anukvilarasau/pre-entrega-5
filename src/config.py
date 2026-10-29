"""
config.py — Configuración central del agente.

Todo lo que puede cambiar (claves, modelo, índice, límites) vive acá, en UN
solo lugar. El resto de los módulos importa estos valores.
"""

import os

# Variables de entorno para torch / sentence-transformers: tienen que definirse
# ANTES de importar esas librerías (por eso config se importa primero).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # las claves se leen del .env (que NO se sube)

DATA_DIR: Path = BASE_DIR / "data"
CHECKPOINTS_DB: Path = DATA_DIR / "checkpoints.sqlite"  # memoria persistente
TRACES_DIR: Path = BASE_DIR / "traces"                  # trazas ReAct (.json y .log)

# ---------------------------------------------------------------------------
# CLAVES Y SERVICIOS (leídos del .env)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5").strip()
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "").strip()
INDEX_NAME: str = os.getenv("INDEX_NAME", "pre-entrega-4").strip()
PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "httpx-docs").strip()

# ---------------------------------------------------------------------------
# EMBEDDINGS — el MISMO modelo con el que se indexó en la Pre-entrega 4
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
EMBEDDING_DEVICE: str = "cpu"  # en la GPU del Mac (MPS) hubo segfaults

# Por debajo de esta similitud (coseno) un fragmento se considera NO relevante.
# Calibrado con consultas reales: temas que SÍ están en la documentación dan
# 0.47-0.64 (timeouts, proxies, HTTP/2, cookies); temas que NO están dan menos
# (websockets 0.43, ORM 0.08). Una consulta en castellano también da bajo
# (0.36), porque el modelo de embeddings y los documentos están en inglés.
UMBRAL_RELEVANCIA: float = 0.45

# ---------------------------------------------------------------------------
# AGENTE
# ---------------------------------------------------------------------------
RECURSION_LIMIT: int = 10          # techo de pasos del grafo por invocación
MAX_TOKENS_HISTORIAL: int = 6000   # si el historial supera esto, se recorta
MAX_TOKENS_RESPUESTA: int = 1500


class ErrorDeConfiguracion(Exception):
    """Falta una clave o un dato del .env: el programa no puede arrancar."""


def validar_entorno() -> None:
    """Verifica que las variables obligatorias estén en el .env."""
    faltan: list[str] = [
        nombre
        for nombre, valor in {
            "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
            "PINECONE_API_KEY": PINECONE_API_KEY,
            "INDEX_NAME": INDEX_NAME,
            "PINECONE_NAMESPACE": PINECONE_NAMESPACE,
        }.items()
        if not valor
    ]
    if faltan:
        raise ErrorDeConfiguracion(
            f"Faltan variables en el .env: {', '.join(faltan)}. "
            "Copiá .env.example como .env y completalas."
        )


def configurar_logs() -> None:
    """Logs con hora y milisegundos; se silencian las librerías ruidosas."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for ruidoso in ("httpx", "httpx2", "httpcore", "urllib3", "anthropic", "huggingface_hub",
                    "sentence_transformers", "transformers", "pinecone", "torch", "aiosqlite"):
        logging.getLogger(ruidoso).setLevel(logging.ERROR)
