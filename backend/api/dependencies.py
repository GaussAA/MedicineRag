"""统一的依赖注入模块

提供所有可注入的依赖项工厂函数，
解决路由层重复定义依赖的问题。
"""

from functools import lru_cache
from typing import Generator

from fastapi import Depends

from rag.core.engine import RAGEngine
from rag.agents.medical_agent import MedicalAgent, AgentConfig, create_medical_agent
from backend.services.qa_service import QAService
from backend.services.doc_service import DocService
from backend.services.security_service import SecurityService
from backend.services.question_type_detector import QuestionTypeDetector
from backend.services.confidence_calculator import ConfidenceCalculator
from backend.statistics import QAStats, get_stats_instance
from backend.logging_config import get_logger
from backend.config import Config

logger = get_logger(__name__)


# ============================================================================
# RAG引擎依赖
# ============================================================================

@lru_cache()
def get_rag_engine() -> RAGEngine:
    """获取RAG引擎实例（单例）
    
    使用lru_cache确保全局只有一个RAGEngine实例，
    避免重复初始化带来的性能开销。
    
    Returns:
        RAGEngine: RAG引擎实例
    """
    logger.info("初始化RAG引擎实例")
    return RAGEngine()


# ============================================================================
# 服务层依赖
# ============================================================================

@lru_cache()
def get_security_service() -> SecurityService:
    """获取安全服务实例（单例）
    
    Returns:
        SecurityService: 安全服务实例
    """
    return SecurityService()


def get_qa_service(
    rag_engine: RAGEngine = None,
    security_service: SecurityService = None
) -> QAService:
    """获取QA服务实例（依赖注入）
    
    如果未传入参数，则自动从依赖容器获取。
    这样可以支持单元测试时手动传入mock对象。
    
    Args:
        rag_engine: RAG引擎实例（可选）
        security_service: 安全服务实例（可选）
        
    Returns:
        QAService: QA服务实例
    """
    if rag_engine is None:
        rag_engine = get_rag_engine()
    if security_service is None:
        security_service = get_security_service()
    return QAService(rag_engine, security_service)


def get_doc_service(rag_engine: RAGEngine = None) -> DocService:
    """获取文档服务实例（依赖注入）
    
    Args:
        rag_engine: RAG引擎实例（可选）
        
    Returns:
        DocService: 文档服务实例
    """
    if rag_engine is None:
        rag_engine = get_rag_engine()
    return DocService(rag_engine)


# ============================================================================
# 统计模块依赖
# ============================================================================

def get_stats() -> QAStats:
    """获取统计模块实例
    
    Returns:
        QAStats: 统计模块实例
    """
    return get_stats_instance()


# ============================================================================
# 配置依赖
# ============================================================================

@lru_cache()
def get_config() -> Config:
    """获取配置实例
    
    Returns:
        Config: 配置实例
    """
    return Config()


# ============================================================================
# FastAPI Depends 兼容的依赖函数
# ============================================================================

def get_rag_engine_dep():
    """FastAPI依赖兼容的RAG引擎获取函数
    
    Returns:
        RAGEngine: RAG引擎实例
    """
    return get_rag_engine()


def get_security_service_dep():
    """FastAPI依赖兼容的安全服务获取函数
    
    Returns:
        SecurityService: 安全服务实例
    """
    return get_security_service()


def get_qa_service_dep(
    rag_engine: RAGEngine = Depends(get_rag_engine_dep),
    security_service: SecurityService = Depends(get_security_service_dep)
) -> QAService:
    """FastAPI依赖兼容的QA服务获取函数
    
    Args:
        rag_engine: RAG引擎实例
        security_service: 安全服务实例
        
    Returns:
        QAService: QA服务实例
    """
    return QAService(rag_engine, security_service)


def get_doc_service_dep(
    rag_engine: RAGEngine = Depends(get_rag_engine_dep)
) -> DocService:
    """FastAPI依赖兼容的文档服务获取函数
    
    Args:
        rag_engine: RAG引擎实例
        
    Returns:
        DocService: 文档服务实例
    """
    return DocService(rag_engine)


# ============================================================================
# Agent 依赖
# ============================================================================

@lru_cache()
def get_question_type_detector() -> QuestionTypeDetector:
    """获取问题类型检测器实例（单例）
    
    Returns:
        QuestionTypeDetector: 问题类型检测器实例
    """
    return QuestionTypeDetector()


@lru_cache()
def get_confidence_calculator() -> ConfidenceCalculator:
    """获取置信度计算器实例（单例）
    
    Returns:
        ConfidenceCalculator: 置信度计算器实例
    """
    return ConfidenceCalculator()


@lru_cache()
def get_medical_agent() -> MedicalAgent:
    """获取医疗问答 Agent 实例（单例）
    
    Returns:
        MedicalAgent: 医疗问答 Agent 实例
    """
    rag_engine = get_rag_engine()
    security_service = get_security_service()
    question_type_detector = get_question_type_detector()
    confidence_calculator = get_confidence_calculator()
    
    config = AgentConfig(
        max_steps=10,
        temperature=0.2,
        timeout=300,
        enable_reflection=True,
        enable_followup=True,
        enable_knowledge_gap=True,
        min_confidence_threshold=0.5
    )
    
    return create_medical_agent(
        rag_engine=rag_engine,
        security_service=security_service,
        question_type_detector=question_type_detector,
        confidence_calculator=confidence_calculator,
        config=config
    )


def get_medical_agent_dep(
    rag_engine: RAGEngine = Depends(get_rag_engine_dep),
    security_service: SecurityService = Depends(get_security_service_dep)
) -> MedicalAgent:
    """FastAPI依赖兼容的 Medical Agent 获取函数
    
    Args:
        rag_engine: RAG引擎实例
        security_service: 安全服务实例
        
    Returns:
        MedicalAgent: 医疗问答 Agent 实例
    """
    return get_medical_agent()


# 需要导入Depends以支持FastAPI依赖注入
__all__ = [
    "get_rag_engine",
    "get_security_service", 
    "get_qa_service",
    "get_doc_service",
    "get_stats",
    "get_config",
    "get_question_type_detector",
    "get_confidence_calculator",
    "get_medical_agent",
    # FastAPI Depends兼容版本
    "get_rag_engine_dep",
    "get_security_service_dep",
    "get_qa_service_dep",
    "get_doc_service_dep",
]
