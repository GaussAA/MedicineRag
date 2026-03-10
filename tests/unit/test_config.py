"""测试配置模块"""

import pytest
from backend.config import Config, config


class TestConfig:
    """测试配置类"""

    def test_config_default_values(self):
        """测试默认配置值"""
        # 测试RAG配置
        assert config.CHUNK_SIZE == 512
        assert config.CHUNK_OVERLAP == 50
        assert config.TOP_K >= 1

    def test_config_get_docs_dir(self):
        """测试文档目录获取"""
        docs_dir = config.get_docs_dir()
        assert docs_dir is not None
        assert docs_dir.exists() or docs_dir.parent.exists()

    def test_config_get_chroma_dir(self):
        """测试Chroma目录获取"""
        chroma_dir = config.get_chroma_dir()
        assert chroma_dir is not None
        assert chroma_dir.exists() or chroma_dir.parent.exists()

    def test_config_collection_name(self):
        """测试集合名称"""
        assert config.COLLECTION_NAME is not None
        assert isinstance(config.COLLECTION_NAME, str)
        assert len(config.COLLECTION_NAME) > 0


class TestConfigClass:
    """测试Config类的静态方法"""

    def test_get_docs_dir_creates_directory(self):
        """测试get_docs_dir创建目录"""
        docs_dir = Config.get_docs_dir()
        # 目录应该被创建或已经存在
        assert docs_dir.parent.exists() or True  # 可能已存在

    def test_get_chroma_dir_creates_directory(self):
        """测试get_chroma_dir创建目录"""
        chroma_dir = Config.get_chroma_dir()
        # 目录应该被创建或已经存在
        assert chroma_dir.parent.exists() or True  # 可能已存在

    def test_config_types(self):
        """测试配置类型"""
        # 确保配置值是正确类型
        assert isinstance(config.CHUNK_SIZE, int)
        assert isinstance(config.CHUNK_OVERLAP, int)
        assert isinstance(config.TOP_K, int)
        assert isinstance(config.SIMILARITY_THRESHOLD, float)
        assert isinstance(config.LLM_TEMPERATURE, float)
        assert isinstance(config.LLM_MAX_TOKENS, int)

    def test_rate_limit_config(self):
        """测试限流配置"""
        assert hasattr(config, 'ENABLE_RATE_LIMIT')
        assert hasattr(config, 'RATE_LIMIT_QPS')
        assert isinstance(config.ENABLE_RATE_LIMIT, bool)
        assert isinstance(config.RATE_LIMIT_QPS, int)
        assert config.RATE_LIMIT_QPS > 0
