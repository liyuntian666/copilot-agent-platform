import json
from typing import List
import chromadb
from tqdm import tqdm

from entity import Parameter, Tool
from models import RemoteEmbeddingModel


class CustomizeMilvus:

    def __init__(self, uri, db_name):
        # Chroma 客户端，数据持久化到本地目录（uri 参数保留但不使用）
        self.chroma_client = chromadb.PersistentClient(path=f"./chroma_data/{db_name}")
        self.embeddingModel = RemoteEmbeddingModel()
        self.db_name = db_name

    def _get_collection_name(self, collection_name):
        # 避免集合名冲突，加上数据库名前缀
        return f"{self.db_name}_{collection_name}"

    def list_collections(self):
        prefix = f"{self.db_name}_"
        return [col.name.replace(prefix, "") for col in self.chroma_client.list_collections() if col.name.startswith(prefix)]

    def has_collection(self, collection_name):
        return collection_name in self.list_collections()

    def load_collection_into_memory(self, collection_name):
        # Chroma 不需要显式加载，保持接口空实现
        pass

    def create_collection(self, collection_name):
        full_name = self._get_collection_name(collection_name)
        try:
            self.chroma_client.create_collection(
                name=full_name,
                embedding_function=None,  # 我们手动提供 embedding
                metadata={"hnsw:space": "cosine"}
            )
        except chromadb.errors.UniqueConstraintError:
            # 集合已存在，忽略
            pass

    def insert_embeddings(self, collection_name, embeddings, datas):
        full_name = self._get_collection_name(collection_name)
        if not self.has_collection(collection_name):
            self.create_collection(collection_name)
        collection = self.chroma_client.get_collection(full_name)

        batch_ids = []
        batch_embeddings = []
        batch_metadatas = []
        batch_documents = []
        for idx, (data, embedding) in enumerate(zip(datas, embeddings)):
            # 使用 tool_id 作为主键（转为字符串）
            tool_id = str(data.get("tool_id", idx))
            batch_ids.append(tool_id)
            # 确保 embedding 是列表
            emb = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            batch_embeddings.append(emb)
            # 元数据：除 embedding 以外的字段
            metadata = {k: v for k, v in data.items() if k != "embedding"}
            batch_metadatas.append(metadata)
            batch_documents.append(data.get("operation_summary", ""))
            # 每 10 条批量写入
            if len(batch_ids) >= 10:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                batch_ids, batch_embeddings, batch_metadatas, batch_documents = [], [], [], []
        # 写入剩余数据
        if batch_ids:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )

    def drop_collection(self, collection_name):
        full_name = self._get_collection_name(collection_name)
        try:
            self.chroma_client.delete_collection(full_name)
        except ValueError:
            pass

    def insert_tools(self, collection_name, tools: List[Tool]):
        if not self.has_collection(collection_name):
            self.create_collection(collection_name)
        embeddings, datas = self.embed_chunked_data(tools)
        self.insert_embeddings(collection_name, embeddings, datas)

    def embed_chunked_data(self, tools: List[Tool]):
        datas = []
        chunk_texts = []
        for tool in tools:
            new_data = {
                "tool_id": tool.tool_id,
                "operation_summary": f"{tool.operationId}: {tool.name_for_human}: {tool.description}",
            }
            datas.append(new_data)
            chunk_texts.append(new_data["operation_summary"])
        embeddings = self.embeddingModel.get_batch_embeddings(chunk_texts)
        return embeddings, datas

    def get_docs(self, collection_name, query, topk=5):
        full_name = self._get_collection_name(collection_name)
        if not self.has_collection(collection_name):
            return []
        collection = self.chroma_client.get_collection(full_name)
        query_embedding = self.embeddingModel.get_embedding(query)
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=topk,
            include=["metadatas"]
        )
        docs = []
        if results['metadatas']:
            for meta in results['metadatas'][0]:
                if meta and "tool_id" in meta:
                    docs.append(meta["tool_id"])
        return docs

    def delete_tools(self, tools: List[Tool]):
        ids = [str(tool.tool_id) for tool in tools]
        full_name = self._get_collection_name("tools")
        if self.has_collection("tools"):
            collection = self.chroma_client.get_collection(full_name)
            collection.delete(ids=ids)

    def get_all_entity(self, ids):
        full_name = self._get_collection_name("tools")
        if not self.has_collection("tools"):
            return []
        collection = self.chroma_client.get_collection(full_name)
        entities = []
        for id_str in ids:
            result = collection.get(ids=[str(id_str)], include=["metadatas", "documents"])
            if result['metadatas'] and len(result['metadatas']) > 0:
                meta = result['metadatas'][0]
                entities.append({
                    "tool_id": meta.get("tool_id"),
                    "operation_summary": result['documents'][0] if result['documents'] else ""
                })
        return entities