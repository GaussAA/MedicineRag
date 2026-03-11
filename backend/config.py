"""配置管理模块"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logger = logging.getLogger(__name__)


def _safe_int(value: str, default: int, name: str) -> int:
    """安全地解析整数配置"""
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"配置项 {name} 值 '{value}' 无效，使用默认值 {default}")
        return default


def _safe_float(value: str, default: float, name: str) -> float:
    """安全地解析浮点数配置"""
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"配置项 {name} 值 '{value}' 无效，使用默认值 {default}")
        return default


def _safe_bool(value: str, default: bool, name: str) -> bool:
    """安全地解析布尔值配置"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        # 空字符串返回默认值
        if not value.strip():
            return default
        return value.lower() in ("true", "1", "yes", "on")
    logger.warning(f"配置项 {name} 值 '{value}' 无效，使用默认值 {default}")
    return default


class Config:
    """系统配置类"""

    # Ollama配置
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:latest")
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "qwen3:8b")
    OLLAMA_RERANK_MODEL: str = os.getenv("OLLAMA_RERANK_MODEL", "dengcao/bge-reranker-v2-m3:latest")

    # RAG配置 - 使用安全解析方法
    CHUNK_SIZE: int = _safe_int(os.getenv("CHUNK_SIZE", ""), 512, "CHUNK_SIZE")
    CHUNK_OVERLAP: int = _safe_int(os.getenv("CHUNK_OVERLAP", ""), 50, "CHUNK_OVERLAP")
    CHUNK_MAX_LENGTH: int = _safe_int(os.getenv("CHUNK_MAX_LENGTH", ""), 2000, "CHUNK_MAX_LENGTH")  # 超过此长度的chunk将被过滤
    TOP_K: int = _safe_int(os.getenv("TOP_K", ""), 5, "TOP_K")
    SIMILARITY_THRESHOLD: float = _safe_float(os.getenv("SIMILARITY_THRESHOLD", ""), 0.3, "SIMILARITY_THRESHOLD")
    
    # Embedding配置
    EMBEDDING_MAX_LENGTH: int = _safe_int(os.getenv("EMBEDDING_MAX_LENGTH", ""), 2000, "EMBEDDING_MAX_LENGTH")  # embedding时的文本截断长度
    
    # 重排序配置
    ENABLE_RERANK: bool = _safe_bool(os.getenv("ENABLE_RERANK", ""), True, "ENABLE_RERANK")
    RERANK_INITIAL_TOP_K: int = _safe_int(os.getenv("RERANK_INITIAL_TOP_K", ""), 20, "RERANK_INITIAL_TOP_K")

    # LLM生成参数
    LLM_TEMPERATURE: float = _safe_float(os.getenv("LLM_TEMPERATURE", ""), 0.2, "LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = _safe_int(os.getenv("LLM_MAX_TOKENS", ""), 1536, "LLM_MAX_TOKENS")

    # 对话历史控制参数
    MAX_HISTORY_TURNS: int = _safe_int(os.getenv("MAX_HISTORY_TURNS", ""), 5, "MAX_HISTORY_TURNS")
    MAX_ANSWER_LENGTH: int = _safe_int(os.getenv("MAX_ANSWER_LENGTH", ""), 300, "MAX_ANSWER_LENGTH")
    MAX_CONTEXT_LENGTH: int = _safe_int(os.getenv("MAX_CONTEXT_LENGTH", ""), 1500, "MAX_CONTEXT_LENGTH")

    # 向量数据库配置
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")

    # 文档存储目录
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "./data/documents")

    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./data/logs/app.log")
    USE_STRUCTURED_LOG: bool = _safe_bool(os.getenv("USE_STRUCTURED_LOG", ""), False, "USE_STRUCTURED_LOG")
    LOG_MAX_BYTES: int = _safe_int(os.getenv("LOG_MAX_BYTES", ""), 10 * 1024 * 1024, "LOG_MAX_BYTES")  # 默认10MB
    LOG_BACKUP_COUNT: int = _safe_int(os.getenv("LOG_BACKUP_COUNT", ""), 5, "LOG_BACKUP_COUNT")

    # 集合名称
    COLLECTION_NAME: str = "medical_knowledge"

    # Embedding缓存配置
    ENABLE_EMBEDDING_CACHE: bool = _safe_bool(os.getenv("ENABLE_EMBEDDING_CACHE", ""), True, "ENABLE_EMBEDDING_CACHE")
    EMBEDDING_CACHE_SIZE: int = _safe_int(os.getenv("EMBEDDING_CACHE_SIZE", ""), 100, "EMBEDDING_CACHE_SIZE")
    EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_CACHE_DIR", "./data/embedding_cache")

    # 分块策略配置
    CHUNK_BY_TITLE: bool = _safe_bool(os.getenv("CHUNK_BY_TITLE", ""), True, "CHUNK_BY_TITLE")
    PRESERVE_MEDICAL_TERMS: bool = _safe_bool(os.getenv("PRESERVE_MEDICAL_TERMS", ""), True, "PRESERVE_MEDICAL_TERMS")
    
    # API限流配置
    ENABLE_RATE_LIMIT: bool = _safe_bool(os.getenv("ENABLE_RATE_LIMIT", ""), True, "ENABLE_RATE_LIMIT")
    RATE_LIMIT_QPS: int = _safe_int(os.getenv("RATE_LIMIT_QPS", ""), 10, "RATE_LIMIT_QPS")
    
    # 各接口限流配置（次/分钟）
    RATE_LIMIT_QA_MAX: int = _safe_int(os.getenv("RATE_LIMIT_QA_MAX", ""), 30, "RATE_LIMIT_QA_MAX")
    RATE_LIMIT_UPLOAD_MAX: int = _safe_int(os.getenv("RATE_LIMIT_UPLOAD_MAX", ""), 10, "RATE_LIMIT_UPLOAD_MAX")
    RATE_LIMIT_OTHER_MAX: int = _safe_int(os.getenv("RATE_LIMIT_OTHER_MAX", ""), 60, "RATE_LIMIT_OTHER_MAX")
    RATE_LIMIT_WINDOW: int = _safe_int(os.getenv("RATE_LIMIT_WINDOW", ""), 60, "RATE_LIMIT_WINDOW")
    
    # 文件上传配置
    MAX_FILE_SIZE_MB: int = _safe_int(os.getenv("MAX_FILE_SIZE_MB", ""), 50, "MAX_FILE_SIZE_MB")  # 单个文件最大大小（MB）
    ENABLE_DUPLICATE_CHECK: bool = _safe_bool(os.getenv("ENABLE_DUPLICATE_CHECK", ""), True, "ENABLE_DUPLICATE_CHECK")  # 是否启用重复文件检测
    
    # LlamaParse配置（用于高精度PDF解析，可选）
    LLAMAPARSE_API_KEY: str = os.getenv("LLAMAPARSE_API_KEY", "")

    @classmethod
    def get_docs_dir(cls) -> Path:
        """获取文档目录路径"""
        docs_dir = Path(cls.DOCUMENTS_DIR)
        docs_dir.mkdir(parents=True, exist_ok=True)
        return docs_dir

    @classmethod
    def get_chroma_dir(cls) -> Path:
        """获取Chroma数据库路径"""
        chroma_dir = Path(cls.CHROMA_PERSIST_DIRECTORY)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        return chroma_dir

    @classmethod
    def get_log_dir(cls) -> Path:
        """获取日志目录路径"""
        log_file = Path(cls.LOG_FILE)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        return log_file.parent


# 导出配置实例
config = Config()
