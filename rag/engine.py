from rag.chunker import IntelligentChunker, create_chunker
from rag.retriever import create_retriever, HybridRetriever
from rag.reranker import create_reranker, Reranker

import os
import json
import hashlib
import threading
import time
from pathlib import Path
from functools import lru_cache
from collections import OrderedDict
from typing import List, Optional, Dict, Any

# 禁用代理 - 确保Ollama本地调用不受代理影响
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,ollama'
os.environ['no_proxy'] = 'localhost,127.0.0.1,ollama'

import chromadb
from pathlib import Path
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
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# 配置日志
logger = get_logger(__name__)


def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """Retry decorator with exponential backoff
    
    Args:
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Backoff multiplier
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after max retries: {e}")
            raise last_exception
        return wrapper
    return decorator


class EmbeddingCache:
    """Embedding缓存 - LRU实现，支持可选的磁盘持久化"""
    
    # 类级别的缓存目录配置
    _cache_dir = None

    def __init__(self, max_size: int = 100, cache_dir: str = None):
        """初始化Embedding缓存
        
        Args:
            max_size: 最大缓存条目数
            cache_dir: 磁盘缓存目录（可选，设为None则只使用内存缓存）
        """
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
        # 磁盘持久化支持
        self._cache_dir = cache_dir or EmbeddingCache._cache_dir
        self._persist_file = None
        self._load_from_disk()
    
    @classmethod
    def set_cache_dir(cls, cache_dir: str):
        """设置全局缓存目录"""
        cls._cache_dir = cache_dir

    def _get_persist_file(self) -> Optional[Path]:
        """获取持久化文件路径"""
        if self._cache_dir:
            cache_path = Path(self._cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            return cache_path / "embedding_cache.json"
        return None

    def _load_from_disk(self):
        """从磁盘加载缓存"""
        self._persist_file = self._get_persist_file()
        if self._persist_file and self._persist_file.exists():
            try:
                with open(self._persist_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = OrderedDict(data.get('cache', {}))
                    self.hits = data.get('hits', 0)
                    self.misses = data.get('misses', 0)
                logger.info(f"从磁盘加载了 {len(self.cache)} 条缓存记录")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")

    def _save_to_disk(self):
        """保存缓存到磁盘"""
        if self._persist_file:
            try:
                data = {
                    'cache': dict(self.cache),
                    'hits': self.hits,
                    'misses': self.misses
                }
                with open(self._persist_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
            except Exception as e:
                logger.warning(f"保存缓存失败: {e}")

    def get(self, key: str) -> Optional[List[float]]:
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: List[float]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        # 优化：每5次写入异步保存一次（提高保存频率，减少进程异常时的数据丢失）
        if len(self.cache) % 5 == 0 and self._persist_file:
            # 使用后台线程保存
            threading.Thread(target=self._save_to_disk, daemon=True).start()

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": f"{hit_rate:.1f}%"}

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class LLMResponseCache:
    """LLM响应缓存 - 基于问题+检索结果哈希的响应缓存"""
    
    # 类级别的缓存配置
    _enabled = False
    _max_size = 50
    
    def __init__(self, max_size: int = 50):
        """初始化LLM响应缓存
        
        Args:
            max_size: 最大缓存条目数
        """
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
    
    @classmethod
    def set_enabled(cls, enabled: bool, max_size: int = 50):
        """设置全局缓存配置"""
        cls._enabled = enabled
        cls._max_size = max_size
    
    def _generate_cache_key(self, query: str, retrieved_docs: List[Any], question_type: str = None) -> str:
        """生成缓存键
        
        缓存键 = 问题哈希 + 检索结果数量 + 前3个文档的哈希
        """
        # 问题哈希
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]
        
        # 文档数量
        doc_count = len(retrieved_docs)
        
        # 前3个文档的内容哈希（用于区分不同检索结果）
        doc_hashes = []
        for doc in retrieved_docs[:3]:
            doc_text = doc.get('text', '')[:500]  # 取前500字符
            doc_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()[:8]
            doc_hashes.append(doc_hash)
        
        # 问题类型（如果有）
        type_suffix = f"_{question_type}" if question_type else ""
        
        return f"{query_hash}_{doc_count}_{'_'.join(doc_hashes)}{type_suffix}"

    def get(self, query: str, retrieved_docs: List[Any], question_type: str = None) -> Optional[str]:
        """获取缓存的LLM响应"""
        if not LLMResponseCache._enabled:
            return None
            
        cache_key = self._generate_cache_key(query, retrieved_docs, question_type)
        
        with self._lock:
            if cache_key in self.cache:
                self.hits += 1
                self.cache.move_to_end(cache_key)
                logger.debug(f"LLM响应缓存命中: {cache_key[:30]}...")
                return self.cache[cache_key]
            self.misses += 1
            return None

    def put(self, query: str, retrieved_docs: List[Any], response: str, question_type: str = None):
        """缓存LLM响应"""
        if not LLMResponseCache._enabled:
            return
            
        cache_key = self._generate_cache_key(query, retrieved_docs, question_type)
        
        with self._lock:
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
            self.cache[cache_key] = response
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits, 
            "misses": self.misses, 
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }

    def clear(self):
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


class RAGEngine:
    """RAG引擎核心类 - 线程安全的单例模式"""
    
    _instance: Optional['RAGEngine'] = None
    _initialized: bool = False
    _init_lock: threading.Lock = threading.Lock()  # 初始化锁

    def __new__(cls):
        """线程安全的单例模式实现 - 双重检查锁定"""
        if cls._instance is None:
            with cls._init_lock:
                # 二次检查，防止其他线程在锁内已初始化
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化 - 线程安全，只执行一次"""
        # 使用实例级锁保护初始化
        if not hasattr(self, '_instance_lock'):
            self._instance_lock = threading.Lock()
        
        with self._instance_lock:
            if RAGEngine._initialized:
                return
            
            # 创建Ollama客户端
            self.ollama_client = OllamaClient(host=config.OLLAMA_BASE_URL)

            # 初始化向量数据库
            self.db = chromadb.PersistentClient(path=str(config.get_chroma_dir()))

            # 初始化embedding缓存（支持持久化）
            if config.ENABLE_EMBEDDING_CACHE:
                # 设置全局缓存目录
                EmbeddingCache.set_cache_dir(config.EMBEDDING_CACHE_DIR)
                self.embedding_cache = EmbeddingCache(
                    max_size=config.EMBEDDING_CACHE_SIZE,
                    cache_dir=config.EMBEDDING_CACHE_DIR
                )
            else:
                self.embedding_cache = None

            # 初始化LLM响应缓存（默认启用，最多缓存50条）
            self.llm_response_cache = LLMResponseCache(max_size=50)

            # 初始化重排序器
            self.reranker = create_reranker(self.ollama_client)

            RAGEngine._initialized = True
            logger.info("RAG引擎初始化完成（线程安全单例模式）")

    @classmethod
    def get_instance(cls) -> 'RAGEngine':
        """获取单例实例的显式方法"""
        return cls()

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching and retry"""
        cache_key = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # Try to get from cache
        if self.embedding_cache:
            cached = self.embedding_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Embedding cache hit: {cache_key[:50]}...")
                return cached

        # Call API
        try:
            response = self.ollama_client.embeddings(
                model=config.OLLAMA_EMBED_MODEL,
                prompt=text
            )
            embedding = response.embedding

            # Store in cache
            if self.embedding_cache:
                self.embedding_cache.put(cache_key, embedding)

            return embedding
        except Exception as e:
            logger.error(f"获取embedding失败: {e}")
            raise

    def add_documents(self, file_path: str) -> Dict[str, Any]:
        """添加文档到知识库 - 简化版"""
        try:
            # 直接读取文件内容
            from pathlib import Path
            file_path_obj = Path(file_path)

            # 根据文件类型读取内容
            content = ""
            if file_path_obj.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path_obj.suffix.lower() == '.pdf':
                # 优先使用 LlamaParse 解析 PDF（更高精度）
                try:
                    from llama_parse import LlamaParse
                    parser = LlamaParse(result_type="text", API_KEY=config.LLAMAPARSE_API_KEY if hasattr(config, 'LLAMAPARSE_API_KEY') else None)
                    documents = parser.load_data(file_path)
                    content = "\n\n".join([doc.text for doc in documents])
                    logger.info("使用 LlamaParse 解析 PDF")
                except ImportError:
                    # LlamaParse 未安装，回退到 SimpleDirectoryReader
                    logger.info("LlamaParse 未安装，使用 SimpleDirectoryReader 解析 PDF")
                    from llama_index.core import SimpleDirectoryReader
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    documents = reader.load_data()
                    content = "\n\n".join([doc.text for doc in documents])
            else:
                # 其他文件类型使用SimpleDirectoryReader
                from llama_index.core import SimpleDirectoryReader
                reader = SimpleDirectoryReader(input_files=[file_path])
                documents = reader.load_data()
                content = "\n\n".join([doc.text for doc in documents])

            if not content.strip():
                return {
                    "status": "error",
                    "doc_count": 0,
                    "message": "文档内容为空"
                }

            logger.info(f"读取到文档内容，长度: {len(content)} 字符")

            # 使用智能分块器
            chunker = create_chunker()
            chunk_objects = chunker.chunk_text(content, file_path)
            
            # 提取文本用于embedding
            chunks = [chunk.text for chunk in chunk_objects]

            logger.info(f"智能分块后得到 {len(chunks)} 个块")

            # 获取collection
            collection = self._get_collection()

            # 批量添加文档（优化版）
            BATCH_SIZE = 100  # 每批处理100条
            batch_ids = []
            batch_embeddings = []
            batch_documents = []
            batch_metadatas = []
            
            for i, chunk_obj in enumerate(chunk_objects):
                chunk_text = chunk_obj.text
                if not chunk_text.strip():
                    continue

                # 过滤超长文本块
                if len(chunk_text) > config.CHUNK_MAX_LENGTH:
                    logger.warning(f"跳过超长文本块 {i}: {len(chunk_text)} 字符")
                    continue

                try:
                    embedding = self._get_embedding(chunk_text)
                    doc_id = f"doc_{i}_{abs(hash(chunk_text)) % 100000}"

                    # 构建metadata
                    metadata = {
                        "file_path": file_path,
                        "chunk_id": i,
                        "char_count": len(chunk_text)
                    }
                    
                    # 添加标题信息
                    if chunk_obj.title:
                        metadata["title"] = chunk_obj.title

                    batch_ids.append(doc_id)
                    batch_embeddings.append(embedding)
                    batch_documents.append(chunk_text)
                    batch_metadatas.append(metadata)

                    # 达到批次大小时批量添加
                    if len(batch_ids) >= BATCH_SIZE:
                        collection.add(
                            ids=batch_ids,
                            embeddings=batch_embeddings,
                            documents=batch_documents,
                            metadatas=batch_metadatas
                        )
                        logger.info(f"批量添加 {len(batch_ids)} 个块到向量库")
                        batch_ids = []
                        batch_embeddings = []
                        batch_documents = []
                        batch_metadatas = []

                except Exception as e:
                    logger.warning(f"处理文档块 {i} 失败: {e}")
                    continue

            # 添加剩余的批次
            if batch_ids:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                logger.info(f"批量添加最后 {len(batch_ids)} 个块到向量库")

            # 刷新embedding缓存到磁盘（确保新生成的embedding被持久化）
            if self.embedding_cache and hasattr(self.embedding_cache, '_save_to_disk'):
                try:
                    self.embedding_cache._save_to_disk()
                    logger.info("Embedding缓存已刷新到磁盘")
                except Exception as e:
                    logger.warning(f"缓存刷新失败: {e}")

            return {
                "status": "success",
                "doc_count": len(chunks),
                "message": f"成功添加 {len(chunks)} 个文档块到知识库"
            }

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return {
                "status": "error",
                "doc_count": 0,
                "message": f"添加文档失败: {str(e)}"
            }

    def retrieve(self, query: str, top_k: int = None, use_hybrid: bool = True) -> List[Any]:
        """检索相关文档 - 混合检索+重排序版
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_hybrid: 是否使用混合检索（向量+关键词）
            
        Returns:
            检索到的文档列表
            
        Raises:
            EmbeddingError: embedding生成失败
            VectorStoreError: 向量存储查询失败
        """
        top_k = top_k or config.TOP_K

        try:
            # 获取collection
            collection = self._get_collection()
            
            # 确定初始检索数量（重排序需要更多文档）
            initial_top_k = self.reranker.initial_top_k if self.reranker.enabled else top_k
            
            # 使用混合检索，初始检索更多文档
            retriever = create_retriever(self.ollama_client, collection)
            docs = retriever.retrieve(
                query=query,
                top_k=initial_top_k,
                use_hybrid=use_hybrid,
                keyword_weight=0.3  # 30%关键词权重
            )

            # 如果启用重排序且检索到足够多的文档，进行重排序
            if self.reranker.enabled and len(docs) > top_k:
                logger.info(f"初始检索完成，找到 {len(docs)} 个文档，开始重排序...")
                docs = self.reranker.rerank(query, docs, top_k)
                logger.info(f"重排序完成，返回 {len(docs)} 个文档")
            elif len(docs) > top_k:
                # 未启用重排序但文档过多，截断
                docs = docs[:top_k]

            logger.info(f"检索完成，找到 {len(docs)} 个相关文档")
            return docs

        except Exception as e:
            logger.error(f"检索失败: {type(e).__name__}: {e}", exc_info=True)
            raise VectorStoreError(f"检索文档失败: {str(e)}") from e

    @retry_with_backoff(max_retries=3, initial_delay=2.0, backoff_factor=2.0)
    def generate(self, query: str, retrieved_docs: List[Any], question_type: str = None) -> str:
        """Generate answer with retry and caching"""
        
        if not retrieved_docs:
            return "Sorry, no relevant information found in the knowledge base."

        # 尝试从缓存获取
        cached_response = self.llm_response_cache.get(query, retrieved_docs, question_type)
        if cached_response:
            logger.info("使用缓存的LLM响应")
            return cached_response

        # Build context
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])

        # Select appropriate prompt template
        if question_type and question_type in ["symptom", "disease", "medication", "examination"]:
            from rag.prompts import QUESTION_TYPE_PROMPTS
            type_prompt = QUESTION_TYPE_PROMPTS.get(question_type, "")
            context_prompt = f"{context}\n\n{type_prompt}"
        else:
            context_prompt = context

        # Construct full prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(context=context_prompt, question=query)}"

        try:
            logger.debug("Starting LLM generation...")
            
            # Call LLM with timeout
            response = self.ollama_client.chat(
                model=config.OLLAMA_LLM_MODEL,
                messages=[{'role': 'user', 'content': full_prompt}],
                options={
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS
                }
            )

            result = response.message.content
            logger.info("LLM生成完成")
            
            # 缓存响应
            self.llm_response_cache.put(query, retrieved_docs, result, question_type)
            
            return result

        except Exception as e:
            logger.error(f"LLM生成失败: {type(e).__name__}: {e}", exc_info=True)
            
            # 优雅降级：返回基于检索结果的 fallback 回答
            fallback_answer = self._generate_fallback_answer(query, retrieved_docs)
            return fallback_answer

    def generate_stream(self, query: str, retrieved_docs: List[Any], question_type: str = None, full_prompt: str = None):
        """流式生成回答
        
        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档列表
            question_type: 问题类型
            full_prompt: 预先构建好的完整prompt（包含历史上下文）
            
        Yields:
            逐字返回的回答内容
            
        Raises:
            LLMException: LLM调用失败
        """
        if not retrieved_docs:
            yield "抱歉，知识库中未找到相关信息。请先上传医疗文档到知识库。"
            return

        # 如果提供了完整prompt，直接使用
        if full_prompt:
            prompt_to_use = full_prompt
        else:
            # 构建上下文
            context = "\n\n".join([doc['text'] for doc in retrieved_docs])

            # 选择合适的prompt模板
            if question_type and question_type in ["symptom", "disease", "medication", "examination"]:
                from rag.prompts import QUESTION_TYPE_PROMPTS
                type_prompt = QUESTION_TYPE_PROMPTS.get(question_type, "")
                context_prompt = f"{context}\n\n{type_prompt}"
            else:
                context_prompt = context

            # 构造完整prompt
            prompt_to_use = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(context=context_prompt, question=query)}"

        try:
            logger.debug("开始调用LLM流式生成...")
            
            # Call LLM
            response = self.ollama_client.chat(
                model=config.OLLAMA_LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt_to_use}],
                options={
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS
                },
                stream=True
            )

            # 逐字yield
            for chunk in response:
                if chunk.message and chunk.message.content:
                    yield chunk.message.content

            logger.info("LLM流式生成完成")

        except Exception as e:
            logger.error(f"LLM流式生成失败: {type(e).__name__}: {e}", exc_info=True)
            raise LLMException(f"流式生成回答失败: {str(e)}") from e

    def build_prompt(self, query: str, history: Optional[List[Dict]] = None) -> str:
        """构建Prompt（含历史上下文）
        
        Args:
            query: 当前问题
            history: 对话历史
            
        Returns:
            包含历史上下文的完整prompt
        """
        # 如果没有历史，直接返回当前问题
        if not history:
            return query
        
        # 配置参数
        max_history_turns = 5  # 最多保留5轮对话
        max_answer_length = 300  # 每个回答最多保留300字符
        
        # 保留最近N轮对话
        recent_history = history[-max_history_turns:] if len(history) > max_history_turns else history
        
        # 构建历史上下文（带长度控制）
        context_parts = []
        total_length = 0
        max_context_length = 1500  # 历史上下文最多1500字符
        
        for h in reversed(recent_history):
            question = h.get('question', '')[:200]  # 问题最多200字符
            answer = h.get('answer', '')[:max_answer_length]  # 回答最多300字符
            
            turn = f"用户：{question}\n助手：{answer}"
            if total_length + len(turn) > max_context_length:
                break
            context_parts.insert(0, turn)
            total_length += len(turn)
        
        if not context_parts:
            return query
        
        context = "\n\n".join(context_parts)
        
        return f"""对话历史：
{context}

当前问题：{query}

请根据以上对话历史和当前问题进行回答。如果当前问题是对之前问题的追问或补充，请结合上下文回答。"""

    def clear_conversation_history(self):
        """清理对话历史（如果RAG引擎有内存中的历史记录）"""
        # 如果未来需要存储内存中的历史，可以在这里清理
        pass

    def get_retrieved_sources(self, retrieved_docs: List[Any]) -> List[Dict]:
        """提取检索到的文档来源信息"""
        sources = []
        for doc in retrieved_docs:
            score_value = doc.get('score')
            if score_value is not None:
                # 转换距离为相似度百分比
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

    def _get_collection(self):
        """获取collection（启用HNSW索引优化）"""
        try:
            return self.db.get_or_create_collection(
                name=config.COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",  # 使用余弦距离
                    "hnsw:construction_ef": 200,  # 构建时的搜索宽度
                    "hnsw:search_ef": 200,  # 查询时的搜索宽度
                    "hnsw:M": 16  # 索引连接的邻居数
                }
            )
        except Exception as e:
            logger.error(f"获取collection失败: {e}")
            raise

    def is_ready(self) -> bool:
        """检查引擎是否就绪"""
        try:
            collection = self._get_collection()
            return collection.count() > 0
        except Exception as e:
            logger.warning(f"检查引擎就绪状态失败: {e}")
            return False

    def get_document_count(self) -> int:
        """获取索引中的文档数量"""
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception as e:
            logger.warning(f"获取文档数量失败: {e}")
            return 0

    def clear_index(self) -> None:
        """清空索引"""
        try:
            self.db.delete_collection(config.COLLECTION_NAME)
            logger.info("索引已清空")
        except Exception as e:
            logger.warning(f"清空索引失败（可能不存在）: {e}")

    def _generate_fallback_answer(self, query: str, retrieved_docs: List[Any]) -> str:
        """生成 fallback 回答（当 LLM 失败时的优雅降级）
        
        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档列表
            
        Returns:
            基于检索结果的 fallback 回答
        """
        if not retrieved_docs:
            return "抱歉，知识库中未找到相关信息。请先上传医疗文档到知识库。"
        
        # 构建参考来源列表
        sources = []
        for i, doc in enumerate(retrieved_docs[:3]):  # 最多使用3个来源
            text = doc.get('text', '')[:200]
            source = doc.get('metadata', {}).get('source', '未知来源')
            sources.append(f"{i+1}. {text}... (来源: {source})")
        
        sources_text = "\n\n".join(sources)
        
        return f"""抱歉，AI生成回答失败。以下是检索到的相关资料，请参考：

{sources_text}

温馨提示：建议稍后重试，如果问题持续存在请联系管理员。"""

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        result = {}
        
        # Embedding缓存统计
        if self.embedding_cache:
            result["embedding_cache"] = self.embedding_cache.get_stats()
        else:
            result["embedding_cache"] = {"hits": 0, "misses": 0, "hit_rate": "N/A", "message": "缓存未启用"}
        
        # LLM响应缓存统计
        if self.llm_response_cache:
            result["llm_response_cache"] = self.llm_response_cache.get_stats()
        else:
            result["llm_response_cache"] = {"hits": 0, "misses": 0, "hit_rate": "N/A", "message": "缓存未启用"}
        
        return result

    @classmethod
    def reset(cls) -> None:
        """重置单例实例（主要用于测试场景）
        
        调用此方法后，下次获取实例将重新初始化
        """
        with cls._init_lock:
            cls._instance = None
            cls._initialized = False
            logger.info("RAG引擎单例已重置")
