from langchain_community.vectorstores import SQLiteVec
from langchain_huggingface import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = SQLiteVec.from_texts(
    texts=["Hola, ¿cómo estás?", "Ketanji Brown Jackson fue nominada."],
    embedding=embedding_function,
    table="prueba",
    db_file="mi_memoria.db"
)

results = db.similarity_search("¿Qué dijo sobre Jackson?")
print(results[0].page_content)
