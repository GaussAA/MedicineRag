"""RAG引擎核心类 - Facade模式

重构后的RAG引擎，采用Facade模式，
内部组合DocumentProcessor、VectorStoreManager、LLMManager等组件。
保留原有API接口以保持向后兼容。
"""

import os
import hashlib
import threading
from pathlib import Path
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
    """RAG引擎核心类 - Facade模式（线程安全单例）

    支持两种使用方式：
    1. 单例模式（默认）：通过 RAGEngine() 获取单例
    2. 非单例模式：通过 RAGEngine.create_instance() 创建独立实例

    组件访问：
    - 通过 get_component() 方法访问内部组件
    - 支持retriever, reranker, llm_manager, vector_store, document_processor, embedding_cache, llm_response_cache
    """

    _instance: Optional['RAGEngine'] = None
    _initialized: bool = False
    _init_lock: threading.Lock = threading.Lock()
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls, force_new: bool = False):
        """线程安全的单例模式实现 - 双重检查锁定

        Args:
            force_new: 强制创建新实例（不推荐用于生产环境，仅用于测试）
        """
        if force_new:
            # 强制创建新实例
            return super().__new__(cls)

        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def create_instance(cls, **kwargs) -> 'RAGEngine':
        """创建新的RAGEngine实例（非单例）

        用于测试或需要多个独立引擎实例的场景。

        Args:
            **kwargs: 可选的组件参数
                - ollama_client: 自定义Ollama客户端
                - vector_store: 自定义向量存储管理器
                - embedding_cache: 自定义Embedding缓存
                - llm_response_cache: 自定义LLM响应缓存
                - document_processor: 自定义文档处理器
                - retriever: 自定义检索器
                - reranker: 自定义重排序器
                - llm_manager: 自定义LLM管理器

        Returns:
            新的RAGEngine实例
        """
        # 创建不触发单例逻辑的实例
        instance = object.__new__(cls)

        # 初始化实例锁
        instance._instance_lock = threading.Lock()

        # 初始化组件（使用提供的或创建新的）
        instance._init_components(**kwargs)

        logger.info("创建了新的RAGEngine实例（非单例）")
        return instance

    def get_component(self, component_name: str):
        """获取内部组件

        Args:
            component_name: 组件名称
                - retriever: 混合检索器
                - reranker: 重排序器
                - llm_manager: LLM管理器
                - vector_store: 向量存储管理器
                - document_processor: 文档处理器
                - embedding_cache: Embedding缓存
                - llm_response_cache: LLM响应缓存
                - ollama_client: Ollama客户端

        Returns:
            组件实例，如果不存在返回None
        """
        component_map = {
            'retriever': self.retriever,
            'reranker': self.reranker,
            'llm_manager': self.llm_manager,
            'vector_store': self.vector_store,
            'document_processor': self.document_processor,
            'embedding_cache': self.embedding_cache,
            'llm_response_cache': self.llm_response_cache,
            'ollama_client': self.ollama_client,
        }
        return component_map.get(component_name)

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

    def _init_components(self, **kwargs) -> None:
        """初始化所有组件（支持依赖注入）

        Args:
            **kwargs: 可选的组件参数，用于自定义注入
                - ollama_client: 自定义Ollama客户端
                - vector_store: 自定义向量存储管理器
                - embedding_cache: 自定义Embedding缓存
                - llm_response_cache: 自定义LLM响应缓存
                - document_processor: 自定义文档处理器
                - retriever: 自定义检索器
                - reranker: 自定义重排序器
                - llm_manager: 自定义LLM管理器
        """
        # 1. Ollama客户端（优先使用传入的）
        self.ollama_client = kwargs.get('ollama_client') or create_ollama_client()

        # 2. 向量存储管理器
        self.vector_store = kwargs.get('vector_store') or create_vector_store_manager()

        # 3. Embedding缓存
        self.embedding_cache = kwargs.get('embedding_cache') or create_embedding_cache()

        # 4. LLM响应缓存
        self.llm_response_cache = kwargs.get('llm_response_cache') or create_llm_response_cache()

        # 5. 文档处理器
        self.document_processor = kwargs.get('document_processor') or create_document_processor()

        # 6. 检索器（需要collection，优先使用传入的）
        if 'retriever' in kwargs:
            self.retriever = kwargs['retriever']
        else:
            self.retriever = create_retriever(
                ollama_client=self.ollama_client,
                collection=self.vector_store.collection
            )

        # 7. 重排序器
        self.reranker = kwargs.get('reranker') or create_reranker(self.ollama_client)

        # 8. LLM管理器（组合缓存）
        self.llm_manager = kwargs.get('llm_manager') or create_llm_manager(
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
            ids = []
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

                # 构建完整的metadata，包含标题和位置信息
                metadata = {
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "chunk_id": i,
                    "char_count": len(chunk_text),
                    "title": chunk.get("title", ""),  # 添加标题信息
                    "position": chunk.get("position", i)  # 添加位置信息
                }
                # 过滤掉空的title
                if not metadata["title"]:
                    del metadata["title"]

                documents.append(chunk_text)
                embeddings.append(embedding)
                metadatas.append(metadata)
                ids.append(doc_id)

                # 批量添加
                if len(documents) >= batch_size:
                    self.vector_store.add_documents(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                    documents = []
                    embeddings = []
                    metadatas = []
                    ids = []

            # 添加剩余批次
            if documents:
                self.vector_store.add_documents(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
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
            metadata = doc.get('metadata', {})
            title = metadata.get('title', '')
            
            # 构建来源信息
            source_name = metadata.get("file_name", metadata.get("source", metadata.get("file_path", "未知")))
            
            # 如果有标题，添加到内容前面
            if title:
                display_content = f"【{title}】{text}"
            else:
                display_content = text
            
            sources.append({
                "content": display_content[:400] + "..." if len(display_content) > 400 else display_content,
                "source": source_name,
                "title": title,  # 添加标题字段
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