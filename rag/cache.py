"""统一缓存接口模块

重构后的缓存模块，提供：
- 统一的缓存抽象接口（CacheProtocol）
- 支持TTL的泛型缓存基类（BaseCache）
- 磁盘持久化混入类（DiskPersistenceMixin）
- LRU + TTL缓存实现（TTLCache）
- 向后兼容的EmbeddingCache、LLMResponseCache、ChunkCache

设计原则：
- 线程安全
- 支持TTL过期
- 支持LRU淘汰
- 统一的持久化接口
"""

import json
import hashlib
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from collections import OrderedDict
from typing import TypeVar, Generic, List, Optional, Dict, Any, Callable
from dataclasses import dataclass

from backend.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


# =============================================================================
# 缓存条目数据结构
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """缓存条目"""
    value: T
    created_at: float  # 创建时间戳
    expires_at: Optional[float] = None  # 过期时间戳（可选）

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


# =============================================================================
# 缓存接口定义
# =============================================================================

class CacheProtocol(ABC, Generic[T]):
    """缓存抽象接口（Protocol）"""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """获取缓存值"""
        pass

    @abstractmethod
    def put(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None表示永不过期
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        pass


# =============================================================================
# 持久化混入类
# =============================================================================

class DiskPersistenceMixin:
    """磁盘持久化混入类

    提供统一的磁盘持久化接口，
    子类只需实现_get_persist_data和_set_persist_data。
    """

    _persist_file: Optional[Path] = None
    _persist_lock: threading.Lock = threading.Lock()

    @property
    @abstractmethod
    def _cache(self) -> OrderedDict:
        """子类必须提供缓存字典"""
        pass

    def _init_persist_file(self, cache_dir: str, filename: str) -> None:
        """初始化持久化文件"""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        DiskPersistenceMixin._persist_file = cache_path / filename

    @abstractmethod
    def _get_persist_data(self) -> Dict[str, Any]:
        """获取需要持久化的数据，子类实现"""
        pass

    @abstractmethod
    def _set_persist_data(self, data: Dict[str, Any]) -> None:
        """从持久化数据恢复，子类实现"""
        pass

    def _load_from_disk(self) -> bool:
        """从磁盘加载缓存"""
        if not DiskPersistenceMixin._persist_file:
            return False

        persist_file = DiskPersistenceMixin._persist_file
        if not persist_file.exists():
            return False

        try:
            with open(persist_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._set_persist_data(data)
            logger.info(f"从磁盘加载缓存成功: {persist_file.name}")
            return True
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return False

    def _save_to_disk(self) -> None:
        """保存缓存到磁盘（线程安全）"""
        if not DiskPersistenceMixin._persist_file:
            return

        with DiskPersistenceMixin._persist_lock:
            try:
                data = self._get_persist_data()
                with open(DiskPersistenceMixin._persist_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"保存缓存失败: {e}")


# =============================================================================
# 统一缓存基类（支持TTL + LRU）
# =============================================================================

class BaseCache(CacheProtocol[T], Generic[T]):
    """统一缓存基类 - 支持TTL过期和LRU淘汰

    特性：
    - 线程安全
    - 支持TTL过期
    - LRU淘汰策略
    - 可选的磁盘持久化
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: Optional[int] = None,
        enable_persist: bool = False,
        persist_file: Optional[Path] = None
    ):
        """初始化缓存

        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 默认TTL（秒），None表示永不过期
            enable_persist: 是否启用磁盘持久化
            persist_file: 持久化文件路径
        """
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = ttl_seconds
        self._enable_persist = enable_persist
        self._persist_file = persist_file

        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

        # 异步保存计数器
        self._save_counter = 0
        self._save_interval = 5

        # 如果启用持久化，尝试加载
        if self._enable_persist and self._persist_file:
            self._load_from_disk()

    def get(self, key: str) -> Optional[T]:
        """获取缓存值（线程安全，自动过期检查）"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # 检查过期
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # 移动到末尾（LRU）
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """设置缓存值（线程安全）"""
        with self._lock:
            # 计算过期时间
            expires_at: Optional[float] = None
            effective_ttl = ttl if ttl is not None else self._default_ttl
            if effective_ttl is not None:
                expires_at = time.time() + effective_ttl

            # 创建缓存条目
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                expires_at=expires_at
            )

            # 如果key已存在，移到末尾
            if key in self._cache:
                self._cache.move_to_end(key)

            self._cache[key] = entry

            # LRU淘汰
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        # 异步保存（优化性能）
        self._save_counter += 1
        if self._save_counter >= self._save_interval and self._enable_persist:
            self._save_counter = 0
            threading.Thread(target=self._save_to_disk, daemon=True).start()

    def delete(self, key: str) -> bool:
        """删除指定缓存项"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def cleanup_expired(self) -> int:
        """清理过期缓存项，返回清理数量"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            # 统计过期项数量
            expired_count = sum(
                1 for entry in self._cache.values()
                if entry.is_expired()
            )

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "cache_size": len(self._cache),
                "max_size": self._max_size,
                "expired_count": expired_count,
                "default_ttl": self._default_ttl
            }

    def keys(self) -> List[str]:
        """获取所有缓存键"""
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> List[T]:
        """获取所有缓存值"""
        with self._lock:
            return [entry.value for entry in self._cache.values()]

    # 持久化方法（子类可重写）
    def _get_persist_data(self) -> Dict[str, Any]:
        """获取需要持久化的数据"""
        return {
            "cache": {
                k: {"v": v.value, "c": v.created_at, "e": v.expires_at}
                for k, v in self._cache.items()
            },
            "hits": self._hits,
            "misses": self._misses
        }

    def _set_persist_data(self, data: Dict[str, Any]) -> None:
        """从持久化数据恢复"""
        cache_data = data.get("cache", {})
        self._cache = OrderedDict()
        for k, v in cache_data.items():
            entry = CacheEntry(
                value=v["v"],
                created_at=v["c"],
                expires_at=v["e"]
            )
            self._cache[k] = entry
        self._hits = data.get("hits", 0)
        self._misses = data.get("misses", 0)

    def _load_from_disk(self) -> bool:
        """从磁盘加载缓存"""
        if not self._persist_file or not self._persist_file.exists():
            return False

        try:
            with open(self._persist_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._set_persist_data(data)
            # 加载后清理过期项
            self.cleanup_expired()
            logger.info(f"从磁盘加载缓存成功: {self._persist_file.name}")
            return True
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return False

    def _save_to_disk(self) -> None:
        """保存缓存到磁盘"""
        if not self._persist_file:
            return

        try:
            data = self._get_persist_data()
            with open(self._persist_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")


# =============================================================================
# 便捷函数：创建TTL缓存
# =============================================================================

def create_ttl_cache(
    max_size: int = 100,
    ttl_seconds: int = 3600,
    persist_file: Optional[str] = None
) -> BaseCache:
    """创建支持TTL的缓存实例

    Args:
        max_size: 最大缓存条目数
        ttl_seconds: 默认过期时间（秒）
        persist_file: 持久化文件路径

    Returns:
        BaseCache实例
    """
    persist_path = Path(persist_file) if persist_file else None
    return BaseCache(
        max_size=max_size,
        ttl_seconds=ttl_seconds,
        enable_persist=persist_path is not None,
        persist_file=persist_path
    )


# =============================================================================
# 向后兼容的缓存实现
# =============================================================================


class EmbeddingCache(BaseCache[List[float]]):
    """Embedding缓存 - 重构为继承BaseCache，支持TTL和LRU

    特性：
    - 继承BaseCache的线程安全、TTL支持、LRU淘汰
    - 可选的磁盘持久化
    - 向后兼容的API
    """

    _cache_dir: str = None  # 类变量用于全局配置

    def __init__(self, max_size: int = 100, cache_dir: str = None, ttl_seconds: int = 86400):
        """初始化Embedding缓存

        Args:
            max_size: 最大缓存条目数
            cache_dir: 磁盘缓存目录（可选，设为None则只使用内存缓存）
            ttl_seconds: 缓存过期时间（秒），默认24小时
        """
        # 设置类变量
        if cache_dir:
            EmbeddingCache._cache_dir = cache_dir

        # 确定持久化文件路径
        persist_file = None
        enable_persist = False
        if EmbeddingCache._cache_dir:
            cache_path = Path(EmbeddingCache._cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            persist_file = cache_path / "embedding_cache.json"
            enable_persist = True

        # 调用父类初始化
        super().__init__(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enable_persist=enable_persist,
            persist_file=persist_file
        )

        # 兼容性别名
        self._save_counter = 0
        self._save_interval = 5

    @classmethod
    def set_cache_dir(cls, cache_dir: str) -> None:
        """设置全局缓存目录"""
        cls._cache_dir = cache_dir

    def put(self, key: str, value: List[float], ttl: Optional[int] = None) -> None:
        """缓存embedding（保持向后兼容，ttl参数可选）"""
        # 保持向后兼容：原有调用不传ttl
        super().put(key, value, ttl)

    # 保持向后兼容的属性
    @property
    def cache(self) -> OrderedDict:
        """获取缓存字典（用于兼容性）"""
        return OrderedDict((k, v.value) for k, v in self._cache.items())

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


class LLMResponseCache(BaseCache[str]):
    """LLM响应缓存 - 重构为继承BaseCache，支持TTL

    特性：
    - 继承BaseCache的线程安全、TTL支持、LRU淘汰
    - 支持基于问题+检索结果的缓存键生成
    - 可配置启用/禁用
    """

    _enabled: bool = False  # 类变量用于全局配置

    def __init__(self, max_size: int = 50, enabled: bool = True, ttl_seconds: int = 3600):
        """初始化LLM响应缓存

        Args:
            max_size: 最大缓存条目数
            enabled: 是否启用缓存
            ttl_seconds: 缓存过期时间（秒），默认1小时
        """
        # 设置类变量
        LLMResponseCache._enabled = enabled

        # 不启用持久化（LLM响应缓存通常不需要持久化）
        super().__init__(
            max_size=max_size,
            ttl_seconds=ttl_seconds if enabled else None,
            enable_persist=False
        )

    @classmethod
    def set_enabled(cls, enabled: bool, max_size: int = 50) -> None:
        """设置全局缓存配置"""
        cls._enabled = enabled

    def get(self, key: str) -> Optional[str]:
        """获取缓存的LLM响应（如果启用）"""
        if not self._enabled:
            return None
        return super().get(key)

    def put(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """缓存LLM响应（如果启用）"""
        if not self._enabled:
            return
        super().put(key, value, ttl)

    @staticmethod
    def generate_cache_key(
        query: str,
        retrieved_docs: List[Any],
        question_type: str = None
    ) -> str:
        """生成缓存键

        缓存键 = 问题哈希 + 检索结果数量 + 前3个文档的哈希

        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档列表
            question_type: 问题类型（可选）

        Returns:
            缓存键字符串
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

    # 保持向后兼容的属性
    @property
    def cache(self) -> OrderedDict:
        """获取缓存字典（用于兼容性）"""
        return self.cache  # BaseCache已实现

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = super().get_stats()
        stats["enabled"] = self._enabled
        return stats


class ChunkCache(BaseCache[List[Any]]):
    """文档分块缓存 - 重构为继承BaseCache，支持TTL

    特性：
    - 继承BaseCache的线程安全、TTL支持、LRU淘汰
    - 基于内容哈希的缓存键
    """

    def __init__(self, max_size: int = 20, ttl_seconds: int = 604800):
        """初始化分块缓存

        Args:
            max_size: 最大缓存文档数
            ttl_seconds: 缓存过期时间（秒），默认7天
        """
        super().__init__(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enable_persist=False  # 分块缓存通常不需要持久化
        )

    @staticmethod
    def generate_cache_key(content: str) -> str:
        """生成分块缓存键

        Args:
            content: 文档内容

        Returns:
            MD5哈希作为缓存键
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()


# =============================================================================
# 缓存工厂函数
# =============================================================================

def create_embedding_cache(
    max_size: int = 500,
    cache_dir: str = None,
    ttl_seconds: int = 86400
) -> EmbeddingCache:
    """创建Embedding缓存实例（工厂函数）

    Args:
        max_size: 最大缓存条目数
        cache_dir: 磁盘缓存目录
        ttl_seconds: 缓存过期时间（秒），默认24小时

    Returns:
        EmbeddingCache实例
    """
    return EmbeddingCache(
        max_size=max_size,
        cache_dir=cache_dir,
        ttl_seconds=ttl_seconds
    )


def create_llm_response_cache(
    max_size: int = 50,
    enabled: bool = True,
    ttl_seconds: int = 3600
) -> LLMResponseCache:
    """创建LLM响应缓存实例（工厂函数）

    Args:
        max_size: 最大缓存条目数
        enabled: 是否启用缓存
        ttl_seconds: 缓存过期时间（秒），默认1小时

    Returns:
        LLMResponseCache实例
    """
    return LLMResponseCache(
        max_size=max_size,
        enabled=enabled,
        ttl_seconds=ttl_seconds
    )


def create_chunk_cache(
    max_size: int = 20,
    ttl_seconds: int = 604800
) -> ChunkCache:
    """创建文档分块缓存实例（工厂函数）

    Args:
        max_size: 最大缓存文档数
        ttl_seconds: 缓存过期时间（秒），默认7天

    Returns:
        ChunkCache实例
    """
    return ChunkCache(
        max_size=max_size,
        ttl_seconds=ttl_seconds
    )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 核心类
    'CacheProtocol',
    'CacheEntry',
    'BaseCache',
    'DiskPersistenceMixin',
    # 缓存实现
    'EmbeddingCache',
    'LLMResponseCache',
    'ChunkCache',
    # 工厂函数
    'create_ttl_cache',
    'create_embedding_cache',
    'create_llm_response_cache',
    'create_chunk_cache',
]
