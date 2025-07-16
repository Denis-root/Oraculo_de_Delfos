import time
import traceback

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.embeddings import init_embeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import sqlite3
from icecream import ic


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "checkpoints" / "checkpoints.sqlite"

# Crear la carpeta si no existe (opcional, para evitar error)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
ic(DB_PATH)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

store = memory(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
        "dims": 1536,                              # Embedding dimensions
        "fields": ["food_preference", "$"]              # Fields to embed
    }
)