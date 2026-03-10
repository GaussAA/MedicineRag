# rag package - RAG核心引擎模块
#
# 本包提供检索增强生成(RAG)系统的核心功能
#
# 子包结构:
#   rag.core      - 核心模块(engine, retriever, reranker, prompts)
#   rag.processing - 处理模块(chunker, document_processor)
#
# 使用示例:
#   from rag.core.engine import RAGEngine
#   from rag.processing.chunker import IntelligentChunker
#   from rag.cache import EmbeddingCache

__version__ = "0.1.4"