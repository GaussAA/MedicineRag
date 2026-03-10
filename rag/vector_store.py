"""向量存储管理模块

封装ChromaDB向量数据库的所有操作，
提供简洁的API接口。
"""

import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import chromadb
from chromadb.types import Collection

from backend.config import config
from backend.logging_config import get_logger
from backend.exceptions import VectorStoreError

logger = get_logger(__name__)

# 批量处理的默认大小
DEFAULT_BATCH_SIZE = 100


class VectorStoreManager:
    """向量存储管理器 - 封装ChromaDB操作"""

    def __init__(self, client: Optional[chromadb.PersistentClient] = None):
        """初始化向量存储管理器

        Args:
            client: ChromaDB客户端，默认创建持久化客户端
        """
        self._lock = threading.Lock()

        # 创建或使用客户端
        if client is None:
            self._client = chromadb.PersistentClient(
                path=str(config.get_chroma_dir())
            )
        else:
            self._client = client

        # 获取集合
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """获取或创建集合（启用HNSW索引优化）"""
        try:
            return self._client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:search_ef": 200,
                    "hnsw:M": 16
                }
            )
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise VectorStoreError(f"创建集合失败: {str(e)}") from e

    @property
    def collection(self) -> Collection:
        """获取集合"""
        return self._collection

    @property
    def client(self) -> chromadb.PersistentClient:
        """获取客户端"""
        return self._client

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> Dict[str, Any]:
        """添加文档到向量存储

        Args:
            documents: 文档文本列表
            embeddings: embedding向量列表
            metadatas: 元数据列表
            ids: 文档ID列表
            batch_size: 批量处理大小

        Returns:
            添加结果

        Raises:
            VectorStoreError: 添加失败
        """
        if not documents or not embeddings:
            return {"status": "warning", "message": "没有文档或embedding需要添加"}

        if len(documents) != len(embeddings):
            raise VectorStoreError("文档数量与embedding数量不匹配")

        try:
            total_added = 0

            # 分批添加
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
                batch_ids = ids[i:i + batch_size] if ids else None

                self._collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

                total_added += len(batch_docs)
                logger.debug(f"已添加 {len(batch_docs)} 个文档到向量库")

            logger.info(f"成功添加 {total_added} 个文档到向量库")
            return {"status": "success", "count": total_added}

        except Exception as e:
            logger.error(f"添加文档到向量库失败: {e}")
            raise VectorStoreError(f"添加文档失败: {str(e)}") from e

    def add_documents_with_progress(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """添加文档（带进度回调）

        Args:
            documents: 文档文本列表
            embeddings: embedding向量列表
            metadatas: 元数据列表
            ids: 文档ID列表
            batch_size: 批量处理大小
            progress_callback: 进度回调函数 (current, total)

        Returns:
            添加结果

        Raises:
            VectorStoreError: 添加失败
        """
        if not documents or not embeddings:
            return {"status": "warning", "message": "没有文档或embedding需要添加"}

        total = len(documents)

        try:
            total_added = 0

            # 分批添加
            for i in range(0, total, batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
                batch_ids = ids[i:i + batch_size] if ids else None

                self._collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

                total_added += len(batch_docs)

                # 调用进度回调
                if progress_callback:
                    progress_callback(total_added, total)

            logger.info(f"成功添加 {total_added} 个文档到向量库")
            return {"status": "success", "count": total_added}

        except Exception as e:
            logger.error(f"添加文档到向量库失败: {e}")
            raise VectorStoreError(f"添加文档失败: {str(e)}") from e

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """查询向量存储

        Args:
            query_embeddings: 查询向量
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            include: 返回字段 ['documents', 'embeddings', 'metadatas', 'distances']

        Returns:
            查询结果

        Raises:
            VectorStoreError: 查询失败
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        try:
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )

            logger.debug(f"查询完成，返回 {len(results.get('documents', [[]])[0])} 个结果")
            return results

        except Exception as e:
            logger.error(f"查询向量库失败: {e}")
            raise VectorStoreError(f"查询失败: {str(e)}") from e

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """获取指定文档

        Args:
            ids: 文档ID列表
            where: 元数据过滤条件
            limit: 返回数量限制
            include: 返回字段

        Returns:
            文档列表

        Raises:
            VectorStoreError: 获取失败
        """
        if include is None:
            include = ["documents", "metadatas", "embeddings"]

        try:
            results = self._collection.get(
                ids=ids,
                where=where,
                limit=limit,
                include=include
            )

            logger.debug(f"获取到 {len(results.get('documents', []))} 个文档")
            return results

        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            raise VectorStoreError(f"获取文档失败: {str(e)}") from e

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        delete_all: bool = False
    ) -> None:
        """删除文档

        Args:
            ids: 文档ID列表
            where: 元数据过滤条件
            delete_all: 是否删除所有

        Raises:
            VectorStoreError: 删除失败
        """
        try:
            self._collection.delete(
                ids=ids,
                where=where,
                where_document=None if not delete_all else {"$exists": True}
            )
            logger.info(f"成功删除文档")

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise VectorStoreError(f"删除文档失败: {str(e)}") from e

    def count(self) -> int:
        """获取文档数量"""
        try:
            return self._collection.count()
        except Exception as e:
            logger.warning(f"获取文档数量失败: {e}")
            return 0

    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """查看前N个文档"""
        try:
            return self._collection.peek(limit=limit)
        except Exception as e:
            logger.warning(f"查看文档失败: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def clear(self) -> None:
        """清空集合"""
        try:
            self._client.delete_collection(config.COLLECTION_NAME)
            self._collection = self._get_or_create_collection()
            logger.info("向量库已清空")
        except Exception as e:
            logger.warning(f"清空向量库失败: {e}")

    def exists(self) -> bool:
        """检查集合是否存在"""
        try:
            return self._collection.count() > 0
        except Exception:
            return False


class VectorStore:
    """向量存储兼容类 - 保持向后兼容"""

    def __init__(self, manager: Optional[VectorStoreManager] = None):
        """初始化向量存储

        Args:
            manager: VectorStoreManager实例
        """
        self._manager = manager or VectorStoreManager()

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """添加文档（兼容旧接口）"""
        self._manager.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_embedding: List[float], top_k: int = 10, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """查询（兼容旧接口）"""
        return self._manager.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )

    def delete(self, ids: List[str]) -> None:
        """删除文档（兼容旧接口）"""
        self._manager.delete(ids=ids)

    def count(self) -> int:
        """获取文档数量"""
        return self._manager.count()


def create_vector_store_manager(
    client: Optional[chromadb.PersistentClient] = None
) -> VectorStoreManager:
    """创建向量存储管理器的工厂函数

    Args:
        client: ChromaDB客户端

    Returns:
        VectorStoreManager 实例
    """
    return VectorStoreManager(client=client)


# 导出
__all__ = [
    'VectorStoreManager',
    'VectorStore',
    'create_vector_store_manager',
]
