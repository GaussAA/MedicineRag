"""统一缓存接口模块

定义缓存抽象接口和实现类，
提供统一的缓存管理和持久化支持。
"""

import json
import hashlib
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from collections import OrderedDict
from typing import TypeVar, Generic, List, Optional, Dict, Any

from backend.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CacheProtocol(ABC, Generic[T]):
    """缓存抽象接口（Protocol）"""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """获取缓存值"""
        pass

    @abstractmethod
    def put(self, key: str, value: T) -> None:
        """设置缓存值"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        pass


class EmbeddingCache(CacheProtocol[List[float]]):
    """Embedding缓存 - LRU实现，支持可选的磁盘持久化"""

    _cache_dir: str = None
    _persist_file: Path = None

    def __init__(self, max_size: int = 100, cache_dir: str = None):
        """初始化Embedding缓存

        Args:
            max_size: 最大缓存条目数
            cache_dir: 磁盘缓存目录（可选，设为None则只使用内存缓存）
        """
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._save_counter = 0
        self._save_interval = 5  # 每5次写入保存一次

        # 磁盘持久化支持
        if cache_dir:
            EmbeddingCache._cache_dir = cache_dir
        self._init_persist_file()
        self._load_from_disk()

    def _init_persist_file(self) -> None:
        """初始化持久化文件"""
        if EmbeddingCache._cache_dir:
            cache_path = Path(EmbeddingCache._cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            EmbeddingCache._persist_file = cache_path / "embedding_cache.json"

    @classmethod
    def set_cache_dir(cls, cache_dir: str) -> None:
        """设置全局缓存目录"""
        cls._cache_dir = cache_dir

    def _load_from_disk(self) -> None:
        """从磁盘加载缓存"""
        if EmbeddingCache._persist_file and EmbeddingCache._persist_file.exists():
            try:
                with open(EmbeddingCache._persist_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._cache = OrderedDict(data.get('cache', {}))
                    self._hits = data.get('hits', 0)
                    self._misses = data.get('misses', 0)
                logger.info(f"从磁盘加载了 {len(self._cache)} 条Embedding缓存记录")
            except Exception as e:
                logger.warning(f"加载Embedding缓存失败: {e}")

    def _save_to_disk(self) -> None:
        """保存缓存到磁盘"""
        if EmbeddingCache._persist_file:
            try:
                data = {
                    'cache': dict(self._cache),
                    'hits': self._hits,
                    'misses': self._misses
                }
                with open(EmbeddingCache._persist_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"保存Embedding缓存失败: {e}")

    def get(self, key: str) -> Optional[List[float]]:
        """获取缓存的embedding"""
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: List[float]) -> None:
        """缓存embedding"""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        # 优化：每N次写入异步保存一次
        self._save_counter += 1
        if self._save_counter >= self._save_interval and EmbeddingCache._persist_file:
            self._save_counter = 0
            threading.Thread(target=self._save_to_disk, daemon=True).start()

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self._cache),
            "max_size": self._max_size
        }

    @property
    def cache(self) -> OrderedDict:
        """获取缓存字典（用于兼容性）"""
        return self._cache

    @property
    def max_size(self) -> int:
        """获取最大缓存大小（用于兼容性）"""
        return self._max_size

    @property
    def hits(self) -> int:
        """获取缓存命中次数（用于兼容性）"""
        return self._hits

    @hits.setter
    def hits(self, value: int) -> None:
        """设置缓存命中次数（用于兼容性）"""
        self._hits = value

    @property
    def misses(self) -> int:
        """获取缓存未命中次数（用于兼容性）"""
        return self._misses

    @misses.setter
    def misses(self, value: int) -> None:
        """设置缓存未命中次数（用于兼容性）"""
        self._misses = value


class LLMResponseCache(CacheProtocol[str]):
    """LLM响应缓存 - 基于问题+检索结果哈希的响应缓存"""

    _enabled: bool = False
    _max_size: int = 50

    def __init__(self, max_size: int = 50, enabled: bool = True):
        """初始化LLM响应缓存

        Args:
            max_size: 最大缓存条目数
            enabled: 是否启用缓存
        """
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._enabled = enabled
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    @classmethod
    def set_enabled(cls, enabled: bool, max_size: int = 50) -> None:
        """设置全局缓存配置"""
        cls._enabled = enabled
        cls._max_size = max_size

    @staticmethod
    def generate_cache_key(
        query: str,
        retrieved_docs: List[Any],
        question_type: str = None
    ) -> str:
        """生成缓存键

        缓存键 = 问题哈希 + 检索结果数量 + 前3个文档的哈希
        """
        # 问题哈希
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]

        # 文档数量
        doc_count = len(retrieved_docs)

        # 前3个文档的内容哈希
        doc_hashes = []
        for doc in retrieved_docs[:3]:
            doc_text = doc.get('text', '')[:500]
            doc_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()[:8]
            doc_hashes.append(doc_hash)

        # 问题类型
        type_suffix = f"_{question_type}" if question_type else ""

        return f"{query_hash}_{doc_count}_{'_'.join(doc_hashes)}{type_suffix}"

    def get(self, key: str) -> Optional[str]:
        """获取缓存的LLM响应"""
        if not self._enabled:
            return None

        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                logger.debug(f"LLM响应缓存命中: {key[:30]}...")
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: str) -> None:
        """缓存LLM响应"""
        if not self._enabled:
            return

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self._cache),
            "max_size": self._max_size,
            "enabled": self._enabled
        }

    @property
    def cache(self) -> OrderedDict:
        """获取缓存字典（用于兼容性）"""
        return self._cache


class ChunkCache(CacheProtocol[List[Any]]):
    """文档分块缓存 - 基于内容哈希"""

    def __init__(self, max_size: int = 20):
        """初始化分块缓存

        Args:
            max_size: 最大缓存文档数
        """
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def generate_cache_key(content: str) -> str:
        """生成分块缓存键"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get(self, key: str) -> Optional[List[Any]]:
        """获取缓存的分块结果"""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: List[Any]) -> None:
        """缓存分块结果"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self._cache),
            "max_size": self._max_size
        }


# 导出
__all__ = [
    'CacheProtocol',
    'EmbeddingCache',
    'LLMResponseCache',
    'ChunkCache',
]
