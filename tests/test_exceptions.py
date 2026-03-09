"""测试自定义异常类"""

import pytest
from backend.exceptions import (
    MedicalRAGException,
    RAGEngineException,
    EmbeddingError,
    VectorStoreError,
    DocumentParseError,
    ChunkingError,
    LLMException,
    ServiceException,
    DocumentNotFoundError,
    UnsupportedFileTypeError,
    KnowledgeBaseEmptyError,
    SecurityException,
    SensitiveContentError,
)


class TestMedicalRAGException:
    """测试基础异常类"""

    def test_basic_exception(self):
        """测试基础异常"""
        exc = MedicalRAGException("测试错误")
        assert str(exc) == "[UNKNOWN_ERROR] 测试错误"
        assert exc.message == "测试错误"
        assert exc.error_code == "UNKNOWN_ERROR"

    def test_exception_with_code(self):
        """测试带错误码的异常"""
        exc = MedicalRAGException("测试错误", "TEST_CODE")
        assert exc.error_code == "TEST_CODE"

    def test_exception_with_details(self):
        """测试带详情的异常"""
        details = {"key": "value"}
        exc = MedicalRAGException("测试错误", details=details)
        assert exc.details == details

    def test_exception_str_with_details(self):
        """测试带详情的异常字符串"""
        details = {"key": "value"}
        exc = MedicalRAGException("测试错误", details=details)
        assert "详情" in str(exc)


class TestRAGEngineExceptions:
    """测试RAG引擎相关异常"""

    def test_embedding_error(self):
        """测试Embedding异常"""
        exc = EmbeddingError("Embedding生成失败")
        assert exc.error_code == "EMBEDDING_ERROR"

    def test_vector_store_error(self):
        """测试向量存储异常"""
        exc = VectorStoreError("向量存储失败")
        assert exc.error_code == "VECTOR_STORE_ERROR"

    def test_document_parse_error(self):
        """测试文档解析异常"""
        exc = DocumentParseError("文档解析失败")
        assert exc.error_code == "DOCUMENT_PARSE_ERROR"

    def test_chunking_error(self):
        """测试分块异常"""
        exc = ChunkingError("分块失败")
        assert exc.error_code == "CHUNKING_ERROR"


class TestLLMExceptions:
    """测试LLM相关异常"""

    def test_llm_exception(self):
        """测试LLM基础异常"""
        exc = LLMException("LLM调用失败")
        assert exc.error_code == "LLM_ERROR"


class TestServiceExceptions:
    """测试服务层异常"""

    def test_document_not_found_error(self):
        """测试文档不存在异常"""
        exc = DocumentNotFoundError("文档不存在")
        assert exc.error_code == "DOCUMENT_NOT_FOUND"

    def test_unsupported_file_type_error(self):
        """测试不支持的文件类型异常"""
        exc = UnsupportedFileTypeError("不支持的文件类型")
        assert exc.error_code == "UNSUPPORTED_FILE_TYPE"

    def test_knowledge_base_empty_error(self):
        """测试知识库为空异常"""
        exc = KnowledgeBaseEmptyError()
        assert exc.error_code == "KNOWLEDGE_BASE_EMPTY"
        assert "知识库" in exc.message


class TestSecurityExceptions:
    """测试安全相关异常"""

    def test_sensitive_content_error(self):
        """测试敏感内容异常"""
        exc = SensitiveContentError("包含敏感内容", "suicide")
        assert exc.error_code == "SENSITIVE_CONTENT"
        assert exc.details.get("category") == "suicide"
