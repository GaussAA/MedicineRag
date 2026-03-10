"""RAG组件工厂函数模块

提供创建RAG系统各组件的工厂函数，
支持依赖注入和测试Mock。
"""

from typing import Optional

from ollama import Client as OllamaClient
import chromadb

from backend.config import config
from rag.processing.chunker import IntelligentChunker
from rag.core.retriever import HybridRetriever
from rag.core.reranker import Reranker
from rag.cache import EmbeddingCache, LLMResponseCache, ChunkCache
from rag.processing.document_processor import DocumentProcessor
from rag.vector_store import VectorStoreManager
from rag.llm_manager import LLMManager


# ============================================================================
# Embedding 相关
# ============================================================================

def create_embedding_cache() -> EmbeddingCache:
    """创建Embedding缓存"""
    if config.ENABLE_EMBEDDING_CACHE:
        EmbeddingCache.set_cache_dir(config.EMBEDDING_CACHE_DIR)
        return EmbeddingCache(
            max_size=config.EMBEDDING_CACHE_SIZE,
            cache_dir=config.EMBEDDING_CACHE_DIR
        )
    return EmbeddingCache(max_size=0)  # 不缓存


def create_ollama_client() -> OllamaClient:
    """创建Ollama客户端"""
    return OllamaClient(host=config.OLLAMA_BASE_URL)


# ============================================================================
# 分块器
# ============================================================================

def create_chunker() -> IntelligentChunker:
    """创建智能分块器"""
    return IntelligentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


# ============================================================================
# 向量存储
# ============================================================================

def create_chroma_client() -> chromadb.PersistentClient:
    """创建ChromaDB客户端"""
    return chromadb.PersistentClient(path=str(config.get_chroma_dir()))


def create_vector_store_manager(
    client: Optional[chromadb.PersistentClient] = None
) -> VectorStoreManager:
    """创建向量存储管理器"""
    if client is None:
        client = create_chroma_client()
    return VectorStoreManager(client=client)


# ============================================================================
# 检索器
# ============================================================================

def create_retriever(
    ollama_client: Optional[OllamaClient] = None,
    collection: Optional[any] = None
) -> HybridRetriever:
    """创建混合检索器"""
    if ollama_client is None:
        ollama_client = create_ollama_client()

    if collection is None:
        vector_store_manager = create_vector_store_manager()
        collection = vector_store_manager.collection

    return HybridRetriever(
        ollama_client=ollama_client,
        collection=collection,
    )


# ============================================================================
# 重排序器
# ============================================================================

def create_reranker(
    ollama_client: Optional[OllamaClient] = None
) -> Reranker:
    """创建重排序器"""
    if ollama_client is None:
        ollama_client = create_ollama_client()
    return Reranker(ollama_client=ollama_client)


# ============================================================================
# 文档处理器
# ============================================================================

def create_document_processor() -> DocumentProcessor:
    """创建文档处理器"""
    chunker = create_chunker()
    chunk_cache = ChunkCache(max_size=20)
    return DocumentProcessor(chunker=chunker, chunk_cache=chunk_cache)


# ============================================================================
# LLM管理器
# ============================================================================

def create_llm_manager(
    ollama_client: Optional[OllamaClient] = None,
    cache: Optional[LLMResponseCache] = None
) -> LLMManager:
    """创建LLM管理器"""
    if ollama_client is None:
        ollama_client = create_ollama_client()

    if cache is None:
        cache = LLMResponseCache(max_size=50, enabled=True)

    return LLMManager(client=ollama_client, cache=cache)


# ============================================================================
# LLM响应缓存
# ============================================================================

def create_llm_response_cache(max_size: int = 50, enabled: bool = True) -> LLMResponseCache:
    """创建LLM响应缓存"""
    cache = LLMResponseCache(max_size=max_size)
    LLMResponseCache.set_enabled(enabled, max_size)
    return cache


# ============================================================================
# 分块缓存
# ============================================================================

def create_chunk_cache(max_size: int = 20) -> ChunkCache:
    """创建分块缓存"""
    return ChunkCache(max_size=max_size)


# ============================================================================
# 导出所有工厂函数
# ============================================================================

__all__ = [
    # 客户端
    'create_ollama_client',
    'create_chroma_client',
    # 缓存
    'create_embedding_cache',
    'create_llm_response_cache',
    'create_chunk_cache',
    # 组件
    'create_chunker',
    'create_retriever',
    'create_reranker',
    'create_document_processor',
    'create_vector_store_manager',
    'create_llm_manager',
]
