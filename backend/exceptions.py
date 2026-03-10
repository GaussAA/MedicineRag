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


# ========== 统一错误码定义 ==========

class ErrorCode:
    """统一错误码定义

    错误码格式: E + 类别(2位) + 序号(2位)
    - 1xxx: 通用错误
    - 2xxx: RAG引擎错误
    - 3xxx: LLM错误
    - 4xxx: 服务层错误
    - 5xxx: 安全相关错误
    - 6xxx: 配置相关错误
    """

    # 通用错误 (1xxx)
    UNKNOWN_ERROR = ("E1001", "未知错误")
    INVALID_REQUEST = ("E1002", "无效请求")
    INTERNAL_ERROR = ("E1003", "内部错误")
    SERVICE_UNAVAILABLE = ("E1004", "服务不可用")

    # RAG引擎错误 (2xxx)
    EMBEDDING_FAILED = ("E2001", "向量嵌入失败")
    RETRIEVAL_FAILED = ("E2002", "文档检索失败")
    VECTOR_STORE_ERROR = ("E2003", "向量存储错误")
    DOCUMENT_PARSE_FAILED = ("E2004", "文档解析失败")
    CHUNKING_FAILED = ("E2005", "文档分块失败")
    RERANK_FAILED = ("E2006", "重排序失败")
    INDEX_REBUILD_FAILED = ("E2007", "索引重建失败")
    INDEX_CLEAR_FAILED = ("E2008", "索引清空失败")

    # LLM错误 (3xxx)
    LLM_FAILED = ("E3001", "LLM生成失败")
    LLM_TIMEOUT = ("E3002", "LLM响应超时")
    LLM_RESPONSE_INVALID = ("E3003", "LLM响应无效")
    PROMPT_BUILD_FAILED = ("E3004", "Prompt构建失败")

    # 服务层错误 (4xxx)
    DOCUMENT_NOT_FOUND = ("E4001", "文档不存在")
    UNSUPPORTED_FILE_TYPE = ("E4002", "不支持的文件类型")
    KNOWLEDGE_BASE_EMPTY = ("E4003", "知识库为空")
    NO_RELEVANT_DOCUMENTS = ("E4004", "未找到相关文档")
    UPLOAD_FAILED = ("E4005", "文件上传失败")
    DELETE_FAILED = ("E4006", "删除失败")

    # 安全相关错误 (5xxx)
    SENSITIVE_CONTENT = ("E5001", "包含敏感内容")
    EMERGENCY_SYMPTOM = ("E5002", "检测到紧急症状")

    # 配置相关错误 (6xxx)
    CONFIG_NOT_FOUND = ("E6001", "配置项不存在")
    CONFIG_VALUE_INVALID = ("E6002", "配置值无效")
    MODEL_NOT_FOUND = ("E6003", "模型未找到")

    @classmethod
    def get_code(cls, error_name: str) -> str:
        """根据错误名称获取错误码"""
        return getattr(cls, error_name, cls.UNKNOWN_ERROR)[0]

    @classmethod
    def get_message(cls, error_name: str) -> str:
        """根据错误名称获取错误消息"""
        return getattr(cls, error_name, cls.UNKNOWN_ERROR)[1]

    @classmethod
    def get_pair(cls, error_name: str) -> tuple:
        """根据错误名称获取错误码和消息对"""
        return getattr(cls, error_name, cls.UNKNOWN_ERROR)
