"""RAG缓存模块单元测试"""

import pytest
from collections import OrderedDict

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbeddingCache:
    """Embedding缓存单元测试"""

    def test_cache_initialization(self):
        """测试缓存初始化"""
        from rag.cache import EmbeddingCache
        
        cache = EmbeddingCache(max_size=10)
        assert cache.max_size == 10
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache.cache) == 0

    def test_cache_put_and_get(self):
        """测试缓存存取"""
        from rag.cache import EmbeddingCache
        
        cache = EmbeddingCache(max_size=3)
        
        # 存入缓存
        cache.put("key1", [1.0, 2.0, 3.0])
        assert len(cache.cache) == 1
        
        # 获取缓存
        result = cache.get("key1")
        assert result == [1.0, 2.0, 3.0]
        assert cache.hits == 1

    def test_cache_miss(self):
        """测试缓存未命中"""
        from rag.cache import EmbeddingCache
        
        cache = EmbeddingCache(max_size=3)
        
        result = cache.get("nonexistent")
        assert result is None
        assert cache.misses == 1

    def test_cache_lru_eviction(self):
        """测试LRU淘汰策略"""
        from rag.cache import EmbeddingCache
        
        cache = EmbeddingCache(max_size=3)
        
        # 存入3个元素
        cache.put("key1", [1.0])
        cache.put("key2", [2.0])
        cache.put("key3", [3.0])
        assert len(cache.cache) == 3
        
        # 存入第4个元素，应该淘汰key1
        cache.put("key4", [4.0])
        assert len(cache.cache) == 3
        assert "key1" not in cache.cache
        assert "key4" in cache.cache

    def test_cache_stats(self):
        """测试缓存统计"""
        from rag.cache import EmbeddingCache
        
        cache = EmbeddingCache(max_size=3)
        
        cache.put("key1", [1.0])
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == "50.0%"

    def test_cache_clear(self):
        """测试缓存清空"""
        from rag.cache import EmbeddingCache
        
        cache = EmbeddingCache(max_size=3)
        cache.put("key1", [1.0])
        cache.hits = 5
        cache.misses = 3
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
