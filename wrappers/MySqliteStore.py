import sqlite3
from langchain_community.vectorstores import SQLiteVec
from sqlite_vec import loadable_path

class SQLiteNamespaceVectorStore:
    def __init__(self, db_path, embedding_function, table="memories", model_dims=None):
        self.db_path = str(db_path)
        self.embedding_function = embedding_function
        self.table = table
        self.model_dims = model_dims
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.enable_load_extension(True)
        self.connection.execute(f"SELECT load_extension('{loadable_path()}')")
        # Crea la tabla si no existe
        self._create_table_if_not_exists()
        self.vector_store = SQLiteVec(
            table=self.table,
            db_file=self.db_path,
            embedding=self.embedding_function,
            connection=self.connection,
        )

    def _create_table_if_not_exists(self):
        # La tabla se maneja por SQLiteVec pero aquí te aseguras si querés custom SQL
        pass  # Por defecto SQLiteVec lo hace

    def add(self, namespace, texts, metadatas=None):
        # Agrega namespace a cada metadata
        metadatas = metadatas or [{} for _ in texts]
        for md in metadatas:
            md["namespace"] = namespace
        return self.vector_store.add_texts(texts, metadatas=metadatas)

    def search(self, namespace, query, k=3):
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"namespace": namespace}
            }
        )
        return retriever.invoke(query)

    def search_with_metadata(self, namespace, query, metadata_filter=None, k=3):
        filtro = {"namespace": namespace}
        if metadata_filter:
            filtro.update(metadata_filter)
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": filtro
            }
        )
        return retriever.invoke(query)

    def delete(self, namespace, ids=None):
        # Elimina por IDs, pero solo si tienen el namespace correcto
        # Si ids es None, elimina TODO el namespace
        if ids:
            docs = self.vector_store.get_by_ids(ids)
            ids_to_delete = [doc.metadata["namespace"] == namespace and doc.metadata.get("id") for doc in docs]
            self.vector_store.delete(ids=ids_to_delete)
        else:
            # Busca todos los IDs de ese namespace y los elimina
            all_docs = self.search(namespace, "", k=1000)
            ids_to_delete = [doc.metadata.get("id") for doc in all_docs]
            self.vector_store.delete(ids=ids_to_delete)

    def update(self, namespace, id, new_text, new_metadata=None):
        # No hay update directo: elimina y vuelve a agregar
        self.delete(namespace, ids=[id])
        md = new_metadata or {}
        md["namespace"] = namespace
        md["id"] = id
        self.add(namespace, [new_text], metadatas=[md])

    def list_namespaces(self):
        # Devuelve todos los namespaces encontrados en la tabla
        cursor = self.connection.execute(
            f"SELECT DISTINCT json_extract(metadata, '$.namespace') as ns FROM {self.table}"
        )
        return [row["ns"] for row in cursor.fetchall() if row["ns"]]

