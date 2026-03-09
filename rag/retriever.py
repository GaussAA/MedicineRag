"""检索优化模块

功能：
1. 混合检索（向量检索 + 关键词检索）
2. 检索结果后处理（去重、重排序、多样性保障）
3. 查询扩展
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
import re
import jieba
import jieba.analyse

from backend.logging_config import get_logger
from backend.config import config

logger = get_logger(__name__)


class QueryExpander:
    """查询扩展器 - 扩展用户查询以提高召回率"""
    
    # 医学同义词词典
    SYNONYMS = {
        "高血压": ["高血压", "高血压病", "血压高", "血压升高", "血压异常"],
        "糖尿病": ["糖尿病", "高血糖", "血糖高", "糖尿病", "血糖异常"],
        "心脏病": ["心脏病", "心血管疾病", "冠心病", "心肌缺血"],
        "感冒": ["感冒", "上呼吸道感染", "流感", "发热"],
        "咳嗽": ["咳嗽", "咳痰", "干咳"],
        "头痛": ["头痛", "头疼", "头晕", "眩晕"],
        "胃痛": ["胃痛", "胃部不适", "腹痛", "肚子疼"],
        "发烧": ["发烧", "发热", "体温升高", "高烧"],
    }
    
    # 常见拼写错误
    TYPOS = {
        "高压血": "高血压",
        "血糖高": "高血糖",
        "心脏": "心脏病",
    }
    
    def expand(self, query: str) -> List[str]:
        """扩展查询
        
        Args:
            query: 原始查询
            
        Returns:
            扩展后的查询列表
        """
        queries = [query]
        
        # 修正拼写错误
        for wrong, correct in self.TYPOS.items():
            if wrong in query:
                queries.append(query.replace(wrong, correct))
        
        # 添加同义词
        for key, synonyms in self.SYNONYMS.items():
            if key in query:
                queries.extend(synonyms)
        
        # 去重
        return list(set(queries))


class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, ollama_client, collection, expander: QueryExpander = None):
        self.ollama_client = ollama_client
        self.collection = collection
        self.expander = expander or QueryExpander()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_hybrid: 是否使用混合检索
            keyword_weight: 关键词权重
            
        Returns:
            检索结果列表
        """
        # 扩展查询
        expanded_queries = self.expander.expand(query)
        logger.info(f"扩展查询: {expanded_queries}")
        
        # 向量检索
        vector_results = self._vector_search(query, top_k * 2)
        
        if not use_hybrid:
            return self._post_process(vector_results, top_k)
        
        # 关键词检索
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # 合并结果
        merged = self._merge_results(
            vector_results,
            keyword_results,
            keyword_weight
        )
        
        # 后处理
        return self._post_process(merged, top_k)
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索"""
        try:
            # 获取query embedding
            embedding = self.ollama_client.embeddings(
                model=config.OLLAMA_EMBED_MODEL,
                prompt=query
            )
            
            # 向量搜索
            results = self.collection.query(
                query_embeddings=[embedding.embedding],
                n_results=top_k
            )
            
            docs = []
            if results and results.get('documents'):
                for i, doc_text in enumerate(results['documents'][0]):
                    if doc_text:
                        # 获取距离分数
                        distance = results['distances'][0][i] if 'distances' in results else 0
                        
                        # 转换为相似度分数（越接近0越好）
                        score = 1 / (1 + distance)
                        
                        docs.append({
                            'text': doc_text,
                            'score': score,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                            'id': results['ids'][0][i] if results.get('ids') else '',
                            'source': 'vector'
                        })
            
            return docs
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """关键词检索（优化版 - 避免全表扫描）"""
        try:
            # 提取查询关键词
            keywords = self._extract_keywords(query)
            if not keywords:
                return []

            logger.debug(f"提取关键词: {keywords}")

            # 限制获取的文档数量，避免内存溢出
            max_fetch = min(top_k * 20, 500)  # 最多获取500个候选

            try:
                # 优化：使用$or组合多个关键词进行过滤
                if len(keywords) > 1:
                    # 使用$or操作符组合多个$contains条件
                    filter_condition = {
                        "$or": [{"$contains": kw} for kw in keywords[:3]]  # 最多使用3个关键词
                    }
                else:
                    # 单关键词直接使用$contains
                    filter_condition = {"$contains": keywords[0]}

                results = self.collection.get(
                    where_document=filter_condition,
                    limit=max_fetch
                )
            except Exception as e:
                # 修复：不要静默回退到全表扫描，使用peek限制数量
                logger.warning(f"关键词过滤失败，回退到限制数量: {e}")
                results = self.collection.peek(limit=max_fetch)

            if not results or not results.get('documents'):
                return []

            # 优化：使用str.count()代替re.findall()，提高效率
            scored_docs = []
            for i, doc_text in enumerate(results['documents']):
                if not doc_text:
                    continue

                # 计算TF-IDF简化版分数
                score = 0
                for keyword in keywords:
                    # 使用str.count代替re.findall，提高效率
                    count = doc_text.count(keyword)
                    if count > 0:
                        # 对数TF + IDF简化
                        score += (1 + count) * 1.0

                if score > 0:
                    scored_docs.append({
                        'text': doc_text,
                        'score': score,
                        'metadata': results['metadatas'][i] if results.get('metadatas') else {},
                        'id': results['ids'][i] if results.get('ids') else '',
                        'source': 'keyword'
                    })

            # 按分数排序
            scored_docs.sort(key=lambda x: x['score'], reverse=True)

            return scored_docs[:top_k]

        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词 - 使用jieba分词
        
        使用jieba的TF-IDF算法提取关键词，比纯正则匹配更智能
        """
        try:
            # 停用词列表
            stop_words = {
                "请问", "什么", "怎么", "如何", "为什么", "有没有",
                "可以", "能够", "应该", "需要", "关于", "我的",
                "这个", "那个", "一些", "哪些", "那样", "这样",
                "时候", "地方", "情况", "问题", "原因", "方法",
                "治疗", "预防", "注意", "建议", "可能", "是否"
            }
            
            # 方法1：使用jieba的TF-IDF提取关键词（更准确）
            try:
                # 获取TF-IDF关键词，最多5个
                tfidf_keywords = jieba.analyse.extract_tags(
                    text, 
                    topK=5, 
                    withWeight=False
                )
                # 过滤停用词
                keywords = [k for k in tfidf_keywords if k not in stop_words and len(k) >= 2]
                if keywords:
                    logger.debug(f"TF-IDF提取关键词: {keywords}")
                    return keywords
            except Exception as e:
                logger.warning(f"TF-IDF关键词提取失败: {e}")
            
            # 方法2：使用jieba分词 + 词性筛选（备用方案）
            try:
                # 使用精确模式分词
                words = jieba.lcut(text)
                
                # 筛选：保留名词(n)、动词(v)、形容词(a)、名词短语
                # 长度>=2的词
                keywords = [
                    w for w in words 
                    if len(w) >= 2 
                    and w not in stop_words
                    and not w.isdigit()
                    and not re.match(r'^[\d\.\,\-\+]+$', w)
                ]
                
                # 去重并返回前5个
                keywords = list(dict.fromkeys(keywords))[:5]
                logger.debug(f"分词提取关键词: {keywords}")
                return keywords
                
            except Exception as e:
                logger.warning(f"分词关键词提取失败: {e}")
            
            # 方法3：备用 - 纯正则匹配（原有逻辑）
            keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
            keywords = [k for k in keywords if k not in stop_words]
            return keywords[:5]
            
        except Exception as e:
            logger.error(f"关键词提取异常: {e}")
            # 备用方案
            return re.findall(r'[\u4e00-\u9fa5]{2,}', text)[:5]
    
    def _merge_results(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """合并向量和关键词检索结果"""
        # 归一化分数
        vector_max = max([r['score'] for r in vector_results]) if vector_results else 1
        keyword_max = max([r['score'] for r in keyword_results]) if keyword_results else 1
        
        # 文档ID到结果的映射
        merged = {}
        
        for r in vector_results:
            doc_id = r.get('id', r['text'][:50])
            normalized_score = r['score'] / vector_max if vector_max > 0 else 0
            
            merged[doc_id] = {
                'text': r['text'],
                'score': normalized_score * (1 - keyword_weight),
                'metadata': r.get('metadata', {}),
                'id': doc_id,
                'sources': ['vector']
            }
        
        for r in keyword_results:
            doc_id = r.get('id', r['text'][:50])
            normalized_score = r['score'] / keyword_max if keyword_max > 0 else 0
            
            if doc_id in merged:
                merged[doc_id]['score'] += normalized_score * keyword_weight
                merged[doc_id]['sources'].append('keyword')
            else:
                merged[doc_id] = {
                    'text': r['text'],
                    'score': normalized_score * keyword_weight,
                    'metadata': r.get('metadata', {}),
                    'id': doc_id,
                    'sources': ['keyword']
                }
        
        # 转换为列表并排序
        result_list = list(merged.values())
        result_list.sort(key=lambda x: x['score'], reverse=True)
        
        return result_list
    
    def _post_process(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """后处理：去重和多样性保障"""
        if not results:
            return []
        
        # 去重：基于文本相似度
        unique_results = []
        seen_texts: Set[str] = set()
        
        for r in results:
            # 简单去重：取前50个字符的hash
            text_preview = r['text'][:50]
            if text_preview in seen_texts:
                continue
            
            seen_texts.add(text_preview)
            unique_results.append(r)
            
            if len(unique_results) >= top_k:
                break
        
        # 多样性保障：如果前几个结果来自同一个文件，尝试添加其他来源
        final_results = []
        file_sources: Dict[str, int] = {}
        
        for r in unique_results:
            file_path = r.get('metadata', {}).get('file_path', 'unknown')
            
            # 限制每个来源的数量
            if file_sources.get(file_path, 0) < 2:
                final_results.append(r)
                file_sources[file_path] = file_sources.get(file_path, 0) + 1
            
            if len(final_results) >= top_k:
                break
        
        return final_results


def create_retriever(ollama_client, collection) -> HybridRetriever:
    """创建检索器实例"""
    return HybridRetriever(
        ollama_client=ollama_client,
        collection=collection,
        expander=QueryExpander()
    )
