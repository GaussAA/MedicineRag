"""RAG引擎核心类 - Facade模式

重构后的RAG引擎，采用Facade模式，
内部组合DocumentProcessor、VectorStoreManager、LLMManager等组件。
保留原有API接口以保持向后兼容。
"""

import os
import hashlib
import threading
from typing import List, Optional, Dict, Any

# 禁用代理 - 确保Ollama本地调用不受代理影响
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,ollama'
os.environ['no_proxy'] = 'localhost,127.0.0.1,ollama'

import chromadb  # 保留导入以支持测试中的patch
from ollama import Client as OllamaClient

from backend.config import config
from backend.logging_config import get_logger
from backend.exceptions import (
    EmbeddingError,
    VectorStoreError,
    DocumentParseError,
    ChunkingError,
    LLMException,
    LLMTTimeoutError,
)

from rag.factory import (
    create_ollama_client,
    create_chroma_client,
    create_chunker,
    create_retriever,
    create_reranker,
    create_embedding_cache,
    create_llm_response_cache,
    create_document_processor,
    create_vector_store_manager,
    create_llm_manager,
)
from rag.cache import EmbeddingCache, LLMResponseCache
from rag.processing.document_processor import DocumentProcessor
from rag.vector_store import VectorStoreManager
from rag.llm_manager import LLMManager
from rag.core.retriever import HybridRetriever
from rag.core.reranker import Reranker

logger = get_logger(__name__)


class RAGEngine:
    """RAG引擎核心类 - Facade模式（线程安全单例）"""

    _instance: Optional['RAGEngine'] = None
    _initialized: bool = False
    _init_lock: threading.Lock = threading.Lock()
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls):
        """线程安全的单例模式实现 - 双重检查锁定"""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化 - 线程安全，只执行一次"""
        if not hasattr(self, '_instance_lock'):
            self._instance_lock = threading.Lock()

        with self._instance_lock:
            if RAGEngine._initialized:
                return

            # 初始化各组件
            self._init_components()

            RAGEngine._initialized = True
            logger.info("RAG引擎初始化完成（Facade模式）")

    def _init_components(self) -> None:
        """初始化所有组件"""
        # 1. Ollama客户端
        self.ollama_client = create_ollama_client()

        # 2. 向量存储管理器
        self.vector_store = create_vector_store_manager()

        # 3. Embedding缓存
        self.embedding_cache = create_embedding_cache()

        # 4. LLM响应缓存
        self.llm_response_cache = create_llm_response_cache()

        # 5. 文档处理器
        self.document_processor = create_document_processor()

        # 6. 检索器（需要collection）
        self.retriever = create_retriever(
            ollama_client=self.ollama_client,
            collection=self.vector_store.collection
        )

        # 7. 重排序器
        self.reranker = create_reranker(self.ollama_client)

        # 8. LLM管理器（组合缓存）
        self.llm_manager = create_llm_manager(
            ollama_client=self.ollama_client,
            cache=self.llm_response_cache
        )

        logger.info("RAG引擎组件初始化完成")

    @classmethod
    def get_instance(cls) -> 'RAGEngine':
        """获取单例实例的显式方法"""
        return cls()

    # =========================================================================
    # 对外API（保持向后兼容）
    # =========================================================================

    def add_documents(self, file_path: str) -> Dict[str, Any]:
        """添加文档到知识库 - Facade接口

        Args:
            file_path: 文件路径

        Returns:
            添加结果
        """
        try:
            # 使用文档处理器处理文件
            result = self.document_processor.process_file(file_path)

            if result["status"] != "success":
                return result

            chunks = result["chunks"]

            # 提取文本和生成embeddings
            documents = []
            embeddings = []
            metadatas = []
            batch_size = 100

            for i, chunk in enumerate(chunks):
                chunk_text = chunk["text"]
                if not chunk_text.strip():
                    continue

                # 过滤超长文本
                if len(chunk_text) > config.CHUNK_MAX_LENGTH:
                    logger.warning(f"跳过超长文本块 {i}: {len(chunk_text)} 字符")
                    continue

                # 生成embedding
                embedding = self._get_embedding(chunk_text)
                doc_id = f"doc_{i}_{abs(hash(chunk_text)) % 100000}"

                metadata = {
                    "file_path": file_path,
                    "chunk_id": i,
                    "char_count": len(chunk_text)
                }
                if chunk.get("title"):
                    metadata["title"] = chunk["title"]

                documents.append(chunk_text)
                embeddings.append(embedding)
                metadatas.append(metadata)

                # 批量添加
                if len(documents) >= batch_size:
                    self.vector_store.add_documents(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                    documents = []
                    embeddings = []
                    metadatas = []

            # 添加剩余批次
            if documents:
                self.vector_store.add_documents(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

            # 刷新embedding缓存
            if hasattr(self.embedding_cache, '_save_to_disk'):
                try:
                    self.embedding_cache._save_to_disk()
                except Exception as e:
                    logger.warning(f"缓存刷新失败: {e}")

            return {
                "status": "success",
                "doc_count": len(chunks),
                "message": f"成功添加 {len(chunks)} 个文档块到知识库"
            }

        except (DocumentParseError, ChunkingError, EmbeddingError, VectorStoreError) as e:
            logger.error(f"添加文档失败: {e}")
            return {"status": "error", "doc_count": 0, "message": str(e)}
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return {"status": "error", "doc_count": 0, "message": str(e)}

    def retrieve(self, query: str, top_k: int = None, use_hybrid: bool = True) -> List[Any]:
        """检索相关文档 - Facade接口

        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_hybrid: 是否使用混合检索

        Returns:
            检索到的文档列表

        Raises:
            EmbeddingError: embedding生成失败
            VectorStoreError: 向量存储查询失败
        """
        top_k = top_k or config.TOP_K

        try:
            # 确定初始检索数量
            initial_top_k = self.reranker.initial_top_k if self.reranker.enabled else top_k

            # 使用检索器
            docs = self.retriever.retrieve(
                query=query,
                top_k=initial_top_k,
                use_hybrid=use_hybrid,
                keyword_weight=0.3
            )

            # 重排序
            if self.reranker.enabled and len(docs) > top_k:
                logger.info(f"初始检索完成，找到 {len(docs)} 个文档，开始重排序...")
                docs = self.reranker.rerank(query, docs, top_k)
                logger.info(f"重排序完成，返回 {len(docs)} 个文档")
            elif len(docs) > top_k:
                docs = docs[:top_k]

            logger.info(f"检索完成，找到 {len(docs)} 个相关文档")
            return docs

        except Exception as e:
            logger.error(f"检索失败: {type(e).__name__}: {e}", exc_info=True)
            raise VectorStoreError(f"检索文档失败: {str(e)}") from e

    def generate(self, query: str, retrieved_docs: List[Any], question_type: str = None) -> str:
        """生成回答 - Facade接口

        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档
            question_type: 问题类型

        Returns:
            生成的文本
        """
        if not retrieved_docs:
            return "抱歉，知识库中未找到相关信息。"

        # 尝试使用LLM管理器生成
        try:
            context = "\n\n".join([doc['text'] for doc in retrieved_docs])
            return self.llm_manager.generate(query, context, question_type)
        except LLMException as e:
            logger.error(f"LLM生成失败: {e}")
            return self._generate_fallback_answer(query, retrieved_docs)

    def generate_stream(self, query: str, retrieved_docs: List[Any], question_type: str = None, full_prompt: str = None):
        """流式生成回答 - Facade接口

        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档
            question_type: 问题类型
            full_prompt: 预先构建的完整prompt

        Yields:
            逐字返回的回答内容

        Raises:
            LLMException: LLM调用失败
        """
        if not retrieved_docs:
            yield "抱歉，知识库中未找到相关信息。请先上传医疗文档到知识库。"
            return

        try:
            context = "\n\n".join([doc['text'] for doc in retrieved_docs])
            yield from self.llm_manager.generate_stream(query, context, question_type, full_prompt)
        except LLMException as e:
            logger.error(f"LLM流式生成失败: {e}")
            raise

    def build_prompt(self, query: str, history: Optional[List[Dict]] = None) -> str:
        """构建Prompt（含历史上下文） - Facade接口"""
        return self.llm_manager.build_prompt_with_history(query, history)

    def get_retrieved_sources(self, retrieved_docs: List[Any]) -> List[Dict]:
        """提取检索到的文档来源信息"""
        sources = []
        for doc in retrieved_docs:
            score_value = doc.get('score')
            if score_value is not None:
                display_score = max(0, min(100, (1 - score_value) * 100))
            else:
                display_score = None

            text = doc.get('text', '')
            sources.append({
                "content": text[:200] + "..." if len(text) > 200 else text,
                "source": doc.get('metadata', {}).get("file_name",
                    doc.get('metadata', {}).get("source",
                    doc.get('metadata', {}).get("file_path", "未知"))),
                "score": display_score
            })
        return sources

    def _get_embedding(self, text: str) -> List[float]:
        """获取embedding（含缓存）"""
        cache_key = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # 尝试从缓存获取
        if self.embedding_cache:
            cached = self.embedding_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Embedding cache hit: {cache_key[:50]}...")
                return cached

        # 调用API
        try:
            response = self.ollama_client.embeddings(
                model=config.OLLAMA_EMBED_MODEL,
                prompt=text
            )
            embedding = response.embedding

            # 存入缓存
            if self.embedding_cache:
                self.embedding_cache.put(cache_key, embedding)

            return embedding
        except Exception as e:
            logger.error(f"获取embedding失败: {e}")
            raise EmbeddingError(f"获取embedding失败: {str(e)}") from e

    def _generate_fallback_answer(self, query: str, retrieved_docs: List[Any]) -> str:
        """生成fallback回答"""
        return self.llm_manager.generate_fallback(query, retrieved_docs)

    # =========================================================================
    # 管理接口
    # =========================================================================

    def _get_collection(self):
        """获取collection（兼容旧接口）"""
        return self.vector_store.collection

    def is_ready(self) -> bool:
        """检查引擎是否就绪"""
        return self.vector_store.exists()

    def get_document_count(self) -> int:
        """获取索引中的文档数量"""
        return self.vector_store.count()

    def clear_index(self) -> None:
        """清空索引"""
        self.vector_store.clear()
        logger.info("索引已清空")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        result = {}

        if self.embedding_cache:
            result["embedding_cache"] = self.embedding_cache.get_stats()
        else:
            result["embedding_cache"] = {"hits": 0, "misses": 0, "hit_rate": "N/A"}

        if self.llm_response_cache:
            result["llm_response_cache"] = self.llm_response_cache.get_stats()
        else:
            result["llm_response_cache"] = {"hits": 0, "misses": 0, "hit_rate": "N/A"}

        return result

    def clear_conversation_history(self):
        """清理对话历史"""
        pass

    @classmethod
    def reset(cls) -> None:
        """重置单例实例"""
        with cls._init_lock:
            cls._instance = None
            cls._initialized = False
            logger.info("RAG引擎单例已重置")


# 保留原有的工厂函数以保持向后兼容
def create_chunker():
    """创建分块器（兼容旧接口）"""
    from rag.factory import create_chunker as _create_chunker
    return _create_chunker()


def create_retriever(ollama_client, collection):
    """创建检索器（兼容旧接口）"""
    from rag.factory import create_retriever as _create_retriever
    return _create_retriever(ollama_client, collection)


def create_reranker(ollama_client):
    """创建重排序器（兼容旧接口）"""
    from rag.factory import create_reranker as _create_reranker
    return _create_reranker(ollama_client)


# 导出
__all__ = [
    'RAGEngine',
    'create_chunker',
    'create_retriever',
    'create_reranker',
    'EmbeddingCache',
    'LLMResponseCache',
]