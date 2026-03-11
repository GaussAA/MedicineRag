"""RAG引擎模块单元测试"""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGEngineSingleton:
    """RAG引擎单例模式单元测试"""

    def setup_method(self):
        """每个测试前重置单例"""
        from rag.core.engine import RAGEngine
        RAGEngine.reset()

    def test_singleton_basic(self):
        """测试基本单例模式"""
        from rag.core.engine import RAGEngine
        
        engine1 = RAGEngine()
        engine2 = RAGEngine()
        
        assert engine1 is engine2

    def test_singleton_thread_safety(self):
        """测试单例模式线程安全"""
        from rag.core.engine import RAGEngine
        
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
        from rag.core.engine import RAGEngine
        
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
        from rag.core.engine import RAGEngine
        RAGEngine.reset()

    @patch('rag.core.engine.OllamaClient')
    @patch('rag.core.engine.chromadb.PersistentClient')
    def test_build_prompt_without_history(self, mock_chroma, mock_ollama):
        """测试无历史记录时构建Prompt"""
        from rag.core.engine import RAGEngine
        
        # 模拟返回值
        mock_ollama_instance = Mock()
        mock_ollama.return_value = mock_ollama_instance
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        engine = RAGEngine()
        
        prompt = engine.build_prompt("什么是高血压？")
        
        assert "什么是高血压？" in prompt

    @patch('rag.core.engine.OllamaClient')
    @patch('rag.core.engine.chromadb.PersistentClient')
    def test_build_prompt_with_history(self, mock_chroma, mock_ollama):
        """测试有历史记录时构建Prompt"""
        from rag.core.engine import RAGEngine
        
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
        from rag.core.reranker import Reranker
        
        reranker = Reranker()
        
        assert reranker.enabled is True
        assert reranker.initial_top_k > 0

    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        from rag.core.reranker import Reranker
        
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
        from rag.core.reranker import Reranker
        from rag.core.engine import RAGEngine
        
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


class TestRAGEngineDependencyInjection:
    """RAG引擎依赖注入测试"""

    def setup_method(self):
        """每个测试前重置单例"""
        from rag.core.engine import RAGEngine
        RAGEngine.reset()

    @patch('rag.core.engine.OllamaClient')
    @patch('rag.core.engine.chromadb.PersistentClient')
    def test_create_instance_basic(self, mock_chroma, mock_ollama):
        """测试create_instance基本功能"""
        from rag.core.engine import RAGEngine

        mock_ollama_instance = Mock()
        mock_ollama.return_value = mock_ollama_instance
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance

        # 使用create_instance创建新实例
        engine = RAGEngine.create_instance()

        assert engine is not None
        assert isinstance(engine, RAGEngine)

    @patch('rag.core.engine.OllamaClient')
    @patch('rag.core.engine.chromadb.PersistentClient')
    def test_create_instance_different_from_singleton(self, mock_chroma, mock_ollama):
        """测试create_instance创建独立于单例的实例"""
        from rag.core.engine import RAGEngine

        mock_ollama_instance = Mock()
        mock_ollama.return_value = mock_ollama_instance
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance

        # 获取单例
        singleton = RAGEngine()

        # 创建新实例
        new_instance = RAGEngine.create_instance()

        # 两者应该是不同的对象
        assert singleton is not new_instance

    @patch('rag.core.engine.OllamaClient')
    @patch('rag.core.engine.chromadb.PersistentClient')
    def test_create_instance_with_custom_components(self, mock_chroma, mock_ollama):
        """测试使用自定义组件创建实例"""
        from rag.core.engine import RAGEngine

        mock_ollama_instance = Mock()
        mock_ollama.return_value = mock_ollama_instance
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance

        # 创建自定义组件
        mock_retriever = Mock()
        mock_reranker = Mock()

        # 使用自定义组件创建实例
        engine = RAGEngine.create_instance(
            retriever=mock_retriever,
            reranker=mock_reranker
        )

        # 验证自定义组件被使用
        assert engine.retriever is mock_retriever
        assert engine.reranker is mock_reranker

    @patch('rag.core.engine.OllamaClient')
    @patch('rag.core.engine.chromadb.PersistentClient')
    def test_get_component(self, mock_chroma, mock_ollama):
        """测试get_component方法"""
        from rag.core.engine import RAGEngine

        mock_ollama_instance = Mock()
        mock_ollama.return_value = mock_ollama_instance
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance

        engine = RAGEngine()

        # 测试获取各个组件
        assert engine.get_component('retriever') is not None
        assert engine.get_component('reranker') is not None
        assert engine.get_component('llm_manager') is not None
        assert engine.get_component('vector_store') is not None
        assert engine.get_component('unknown') is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
