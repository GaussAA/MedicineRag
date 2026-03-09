"""自定义异常类模块"""

from typing import Optional, Any, Dict


class MedicalRAGException(Exception):
    """基础异常类"""
    
    DEFAULT_CODE = "UNKNOWN_ERROR"
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        # 如果没有传入error_code，尝试从类属性获取
        self.error_code = error_code or getattr(self.__class__, 'DEFAULT_CODE', 'UNKNOWN_ERROR')
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"[{self.error_code}] {self.message} (详情: {self.details})"
        return f"[{self.error_code}] {self.message}"


# ========== RAG引擎相关异常 ==========

class RAGEngineException(MedicalRAGException):
    """RAG引擎基础异常"""
    DEFAULT_CODE = "RAG_ENGINE_ERROR"


class EmbeddingError(RAGEngineException):
    """Embedding生成异常"""
    DEFAULT_CODE = "EMBEDDING_ERROR"


class VectorStoreError(RAGEngineException):
    """向量存储异常"""
    DEFAULT_CODE = "VECTOR_STORE_ERROR"


class DocumentParseError(RAGEngineException):
    """文档解析异常"""
    DEFAULT_CODE = "DOCUMENT_PARSE_ERROR"


class ChunkingError(RAGEngineException):
    """文档分块异常"""
    DEFAULT_CODE = "CHUNKING_ERROR"


# ========== LLM相关异常 ==========

class LLMException(MedicalRAGException):
    """LLM基础异常"""
    DEFAULT_CODE = "LLM_ERROR"


class LLMTTimeoutError(LLMException):
    """LLM调用超时"""
    DEFAULT_CODE = "LLM_TIMEOUT"


class LLMResponseError(LLMException):
    """LLM响应异常"""
    DEFAULT_CODE = "LLM_RESPONSE_ERROR"


# ========== 服务层异常 ==========

class ServiceException(MedicalRAGException):
    """服务层基础异常"""
    DEFAULT_CODE = "SERVICE_ERROR"


class DocumentNotFoundError(ServiceException):
    """文档不存在"""
    DEFAULT_CODE = "DOCUMENT_NOT_FOUND"


class UnsupportedFileTypeError(ServiceException):
    """不支持的文件类型"""
    DEFAULT_CODE = "UNSUPPORTED_FILE_TYPE"


class KnowledgeBaseEmptyError(ServiceException):
    """知识库为空"""
    DEFAULT_CODE = "KNOWLEDGE_BASE_EMPTY"

    def __init__(self, message: str = "知识库为空，请先上传文档", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, None, details)


class NoRelevantDocumentsError(ServiceException):
    """未找到相关文档"""
    DEFAULT_CODE = "NO_RELEVANT_DOCUMENTS"

    def __init__(self, message: str = "未找到相关内容", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, None, details)


# ========== 安全相关异常 ==========

class SecurityException(MedicalRAGException):
    """安全基础异常"""
    DEFAULT_CODE = "SECURITY_ERROR"


class SensitiveContentError(SecurityException):
    """敏感内容"""
    DEFAULT_CODE = "SENSITIVE_CONTENT"

    def __init__(self, message: str, category: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if category:
            details["category"] = category
        super().__init__(message, None, details)


class EmergencySymptomError(SecurityException):
    """紧急症状"""
    DEFAULT_CODE = "EMERGENCY_SYMPTOM"


# ========== 配置相关异常 ==========

class ConfigException(MedicalRAGException):
    """配置基础异常"""
    DEFAULT_CODE = "CONFIG_ERROR"


class ConfigNotFoundError(ConfigException):
    """配置项不存在"""
    DEFAULT_CODE = "CONFIG_NOT_FOUND"


class ConfigValueError(ConfigException):
    """配置值无效"""
    DEFAULT_CODE = "CONFIG_VALUE_ERROR"
