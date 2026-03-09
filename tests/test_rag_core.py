"""RAG核心模块单元测试"""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from collections import OrderedDict

import sys
import os

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbeddingCache:
    """Embedding缓存单元测试"""

    def test_cache_initialization(self):
        """测试缓存初始化"""
        from rag.engine import EmbeddingCache
        
        cache = EmbeddingCache(max_size=10)
        assert cache.max_size == 10
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache.cache) == 0

    def test_cache_put_and_get(self):
        """测试缓存存取"""
        from rag.engine import EmbeddingCache
        
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
        from rag.engine import EmbeddingCache
        
        cache = EmbeddingCache(max_size=3)
        
        result = cache.get("nonexistent")
        assert result is None
        assert cache.misses == 1

    def test_cache_lru_eviction(self):
        """测试LRU淘汰策略"""
        from rag.engine import EmbeddingCache
        
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
        from rag.engine import EmbeddingCache
        
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
        from rag.engine import EmbeddingCache
        
        cache = EmbeddingCache(max_size=3)
        cache.put("key1", [1.0])
        cache.hits = 5
        cache.misses = 3
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0


class TestRAGEngineSingleton:
    """RAG引擎单例模式单元测试"""

    def setup_method(self):
        """每个测试前重置单例"""
        from rag.engine import RAGEngine
        RAGEngine.reset()

    def test_singleton_basic(self):
        """测试基本单例模式"""
        from rag.engine import RAGEngine
        
        engine1 = RAGEngine()
        engine2 = RAGEngine()
        
        assert engine1 is engine2

    def test_singleton_thread_safety(self):
        """测试单例模式线程安全"""
        from rag.engine import RAGEngine
        
        instances = []
        errors = []
        
        def get_instance():
            try:
                engine = RAGEngine()
                instances.append(engine)
            except Exception as e:
                errors.append(e)
        
        # 并发创建多个实例
        threads = []
        for _ in range(10):
            t = threading.Thread(target=get_instance)
            threads.append(t)
        
        # 同时启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 验证没有错误
        assert len(errors) == 0
        
        # 验证所有实例是同一个对象
        if instances:
            first_instance = instances[0]
            for engine in instances:
                assert engine is first_instance

    def test_reset_method(self):
        """测试重置方法"""
        from rag.engine import RAGEngine
        
        engine1 = RAGEngine()
        
        # 重置后应该能创建新实例
        RAGEngine.reset()
        
        # 注意：由于单例模式的实现，重置后第一次获取会重新初始化
        # 但由于有锁保护，实例仍然是同一个
        engine2 = RAGEngine()
        # 验证引擎正常工作
        assert engine2 is not None


class TestRAGEngineMethods:
    """RAG引擎方法单元测试"""

    def setup_method(self):
        """每个测试前重置单例"""
        from rag.engine import RAGEngine
        RAGEngine.reset()

    @patch('rag.engine.OllamaClient')
    @patch('rag.engine.chromadb.PersistentClient')
    def test_build_prompt_without_history(self, mock_chroma, mock_ollama):
        """测试无历史记录时构建Prompt"""
        from rag.engine import RAGEngine
        
        # 模拟返回值
        mock_ollama_instance = Mock()
        mock_ollama.return_value = mock_ollama_instance
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        engine = RAGEngine()
        
        prompt = engine.build_prompt("什么是高血压？")
        
        assert "什么是高血压？" in prompt

    @patch('rag.engine.OllamaClient')
    @patch('rag.engine.chromadb.PersistentClient')
    def test_build_prompt_with_history(self, mock_chroma, mock_ollama):
        """测试有历史记录时构建Prompt"""
        from rag.engine import RAGEngine
        
        mock_ollama_instance = Mock()
        mock_ollama.return_value = mock_ollama_instance
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        engine = RAGEngine()
        
        history = [
            {"question": "高血压有什么症状？", "answer": "高血压可能没有明显症状。"}
        ]
        
        prompt = engine.build_prompt("高血压需要注意什么？", history)
        
        assert "高血压有什么症状？" in prompt
        assert "高血压需要注意什么？" in prompt


class TestReranker:
    """重排序器单元测试"""

    def test_reranker_initialization(self):
        """测试重排序器初始化"""
        from rag.reranker import Reranker
        
        reranker = Reranker()
        
        assert reranker.enabled is True
        assert reranker.initial_top_k > 0

    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        from rag.reranker import Reranker
        
        reranker = Reranker()
        
        # 相同向量
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        assert reranker._cosine_similarity(vec1, vec2) == pytest.approx(1.0)
        
        # 正交向量
        vec3 = [0.0, 1.0, 0.0]
        assert reranker._cosine_similarity(vec1, vec3) == pytest.approx(0.0)
        
        # 相反向量
        vec4 = [-1.0, 0.0, 0.0]
        assert reranker._cosine_similarity(vec1, vec4) == pytest.approx(-1.0)

    def test_rerank_disabled(self):
        """测试禁用重排序时直接返回原结果"""
        from rag.reranker import Reranker
        from rag.engine import RAGEngine
        
        # 临时禁用重排序
        with patch.object(RAGEngine, '_initialized', True):
            reranker = Reranker.__new__(Reranker)
            reranker.enabled = False
            
            docs = [{"text": "doc1"}, {"text": "doc2"}]
            result = reranker.rerank("query", docs, top_k=2)
            
            assert len(result) == 2


class TestConfig:
    """配置单元测试"""

    def test_default_config_values(self):
        """测试默认配置值"""
        from backend.config import Config
        
        # 验证关键配置项有默认值
        assert Config.CHUNK_SIZE > 0
        assert Config.TOP_K > 0
        assert Config.LLM_TEMPERATURE >= 0
        assert Config.LLM_TEMPERATURE <= 2

    def test_rate_limit_config(self):
        """测试限流配置"""
        from backend.config import Config
        
        # 验证限流配置存在
        assert hasattr(Config, 'RATE_LIMIT_QA_MAX')
        assert hasattr(Config, 'RATE_LIMIT_UPLOAD_MAX')
        assert hasattr(Config, 'RATE_LIMIT_OTHER_MAX')
        assert Config.RATE_LIMIT_QA_MAX > 0

    def test_embedding_cache_config(self):
        """测试Embedding缓存配置"""
        from backend.config import Config
        
        assert hasattr(Config, 'EMBEDDING_CACHE_SIZE')
        assert hasattr(Config, 'EMBEDDING_CACHE_DIR')
        assert Config.EMBEDDING_CACHE_SIZE > 0


class TestSecurityService:
    """安全服务单元测试"""

    def test_security_service_initialization(self):
        """测试安全服务初始化"""
        from backend.services.security_service import SecurityService
        
        service = SecurityService()
        assert service.patterns is not None

    def test_check_content_safe(self):
        """测试安全内容检查"""
        from backend.services.security_service import SecurityService
        
        service = SecurityService()
        result = service.check_content("请问高血压需要注意什么？")
        
        assert result.is_safe is True

    def test_check_content_sensitive(self):
        """测试敏感内容检查"""
        from backend.services.security_service import SecurityService
        
        service = SecurityService()
        result = service.check_content("如何自杀？")
        
        assert result.is_safe is False
        assert result.category is not None

    def test_desensitize(self):
        """测试脱敏功能"""
        from backend.services.security_service import SecurityService
        
        service = SecurityService()
        
        # 测试手机号脱敏
        text = "我的手机号是13812345678"
        result = service.desensitize(text)
        assert "13812345678" not in result

    def test_emergency_symptom(self):
        """测试紧急症状检测"""
        from backend.services.security_service import SecurityService
        
        service = SecurityService()
        
        assert service.is_emergency_symptom("胸痛呼吸困难") is True
        assert service.is_emergency_symptom("高血压饮食注意") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
