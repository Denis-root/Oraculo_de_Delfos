from huggingface_hub import snapshot_download
import shutil
from pathlib import Path
from icecream import ic

# 1. Descarga el modelo en la caché oficial de HuggingFace
local_model_dir = snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2")
ic("Ruta real del modelo:", local_model_dir)

# 2. Copia la carpeta a tu destino personalizado (opcional)
DEST = Path(__file__).resolve().parent/ "models" / "sentence-transformers" / "all-MiniLM-L6-v2"
ic(DEST)
shutil.copytree(local_model_dir, DEST, dirs_exist_ok=True)
ic("¡Modelo copiado a:", DEST)

# 3. Usa el modelo en LangChain
from langchain_huggingface import HuggingFaceEmbeddings
embedding_function = HuggingFaceEmbeddings(model_name=str(DEST))
ic("Embedding cargado correctamente:", embedding_function)
