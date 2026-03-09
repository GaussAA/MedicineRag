"""重排序模块 - 使用双塔模型对检索结果进行重排序

注意：由于Ollama Python客户端没有原生rerank API，这里使用双塔模型（bi-encoder）方式：
- 分别获取查询和文档的embedding
- 计算余弦相似度作为相关性分数
- 按相关性排序

这种方法适用于BGE-M3等支持embedding的模型。
如果使用专门的Cross-Encoder模型(如BGE-Reranker)，建议在Ollama支持rerank API后更新。
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from ollama import Client as OllamaClient

from backend.config import config
from backend.logging_config import get_logger

# 禁用代理
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,ollama'
os.environ['no_proxy'] = 'localhost,127.0.0.1,ollama'

logger = get_logger(__name__)


class Reranker:
    """重排序器 - 使用双塔模型（bi-encoder）方式对检索结果进行重新排序
    
    工作原理：
    1. 获取查询的embedding向量
    2. 并发获取每个文档的embedding向量
    3. 计算查询与每个文档的余弦相似度
    4. 按相似度从高到低排序
    
    注意：如果专门的rerank模型不可用，会自动fallback到embedding模型。
    """
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """初始化重排序器
        
        Args:
            ollama_client: Ollama 客户端实例，如果为 None 则创建新实例
        """
        self.ollama_client = ollama_client or OllamaClient(host=config.OLLAMA_BASE_URL)
        self.model = config.OLLAMA_RERANK_MODEL
        self.enabled = config.ENABLE_RERANK
        self.initial_top_k = config.RERANK_INITIAL_TOP_K
        # fallback模型：使用embedding模型
        self.fallback_model = config.OLLAMA_EMBED_MODEL
        self._use_fallback = False
        
        logger.info(f"重排序器初始化: model={self.model}, fallback={self.fallback_model}, enabled={self.enabled}, initial_top_k={self.initial_top_k}")
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的 embedding 向量
        
        优先使用rerank模型，如果失败则fallback到embedding模型。
        
        Args:
            text: 输入文本
            
        Returns:
            embedding 向量，失败返回 None
        """
        # 截断过长的文本
        max_length = config.EMBEDDING_MAX_LENGTH
        if len(text) > max_length:
            text = text[:max_length]
        
        # 尝试使用rerank模型
        model_to_use = self.model if not self._use_fallback else self.fallback_model
        
        try:
            response = self.ollama_client.embeddings(
                model=model_to_use,
                prompt=text
            )
            return response.embedding
        except Exception as e:
            # 如果使用rerank模型失败，记录并尝试fallback
            if not self._use_fallback:
                logger.warning(f"使用rerank模型 {self.model} 获取embedding失败: {e}，尝试fallback到 {self.fallback_model}")
                self._use_fallback = True
                try:
                    response = self.ollama_client.embeddings(
                        model=self.fallback_model,
                        prompt=text
                    )
                    return response.embedding
                except Exception as fallback_error:
                    logger.error(f"Fallback模型 {self.fallback_model} 也失败: {fallback_error}")
                    return None
            logger.error(f"获取embedding失败: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度 (0-1)
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """对文档列表进行重排序
        
        使用双塔模型计算查询与每个文档的相关性分数，
        然后按相关性从高到低排序。
        
        Args:
            query: 查询文本
            documents: 文档列表，每个文档包含 'text' 和可选的 'score', 'metadata' 字段
            top_k: 返回的 top_k 数量，默认使用配置中的 TOP_K
            
        Returns:
            重排序后的文档列表
        """
        if not self.enabled:
            logger.debug("重排序功能未启用，直接返回原始结果")
            return documents
        
        if not documents:
            logger.warning("重排序输入为空列表")
            return documents
        
        top_k = top_k or config.TOP_K
        
        # 如果文档数量少于等于 top_k，直接返回
        if len(documents) <= top_k:
            logger.debug(f"文档数量 ({len(documents)}) <= top_k ({top_k})，无需重排序")
            return documents
        
        try:
            logger.info(f"开始重排序: query长度={len(query)}, 文档数={len(documents)}, top_k={top_k}")
            
            # 获取查询的 embedding
            query_embedding = self._get_embedding(query)
            
            # 如果获取embedding失败，返回原始结果
            if query_embedding is None:
                logger.warning("获取查询embedding失败，返回原始检索结果")
                return documents[:top_k]
            
            # 使用线程池并发获取文档embeddings
            doc_texts = [(i, doc.get('text', '')) for i, doc in enumerate(documents) if doc.get('text')]
            
            # 并发获取文档embeddings
            doc_embeddings = {}
            max_workers = min(8, len(doc_texts))
            
            def get_doc_embedding(idx_text):
                idx, text = idx_text
                embedding = self._get_embedding(text)
                return idx, embedding
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(get_doc_embedding, item): item[0] for item in doc_texts}
                for future in as_completed(futures):
                    try:
                        idx, embedding = future.result()
                        if embedding is not None:
                            doc_embeddings[idx] = embedding
                    except Exception as e:
                        logger.warning(f"获取文档embedding失败: {e}")
            
            # 如果没有成功获取任何文档embedding，返回原始结果
            if not doc_embeddings:
                logger.warning("没有成功获取任何文档embedding，返回原始检索结果")
                return documents[:top_k]
            
            # 计算每个文档与查询的相关性分数
            scored_docs = []
            for i, doc in enumerate(documents):
                text = doc.get('text', '')
                if not text or i not in doc_embeddings:
                    continue
                
                doc_embedding = doc_embeddings[i]
                
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                # 创建新的文档对象，保留原始信息
                new_doc = doc.copy()
                new_doc['rerank_score'] = float(similarity)
                new_doc['original_index'] = i
                
                # 如果有原始分数，保留并与重排序分数结合
                if 'score' in doc:
                    # 综合分数：重排序分数占 70%，原始分数占 30%
                    original_similarity = 1 - doc.get('score', 0)
                    new_doc['combined_score'] = similarity * 0.7 + original_similarity * 0.3
                
                scored_docs.append(new_doc)
            
            # 按重排序分数降序排序
            scored_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            # 返回 top_k 个文档
            reranked = scored_docs[:top_k]
            
            logger.info(f"重排序完成，返回 {len(reranked)} 个文档，最高分数: {reranked[0].get('rerank_score', 0):.4f}")
            return reranked
            
        except Exception as e:
            logger.error(f"重排序失败: {type(e).__name__}: {e}", exc_info=True)
            # 重排序失败时返回原始结果
            logger.warning("重排序失败，返回原始检索结果")
            return documents[:top_k]
    
    def rerank_with_scores(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """带分数的重排序 - 返回带有相关性分数的文档列表
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的 top_k 数量
            
        Returns:
            重排序后的文档列表，每个文档包含 'rerank_score' 字段
        """
        return self.rerank(query, documents, top_k)


def create_reranker(ollama_client: Optional[OllamaClient] = None) -> Reranker:
    """创建重排序器的工厂函数
    
    Args:
        ollama_client: Ollama 客户端实例
        
    Returns:
        Reranker 实例
    """
    return Reranker(ollama_client)
