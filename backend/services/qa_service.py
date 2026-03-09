"""问答服务模块 - 优化版，增加问题类型识别和统计"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from rag.engine import RAGEngine
from backend.services.security_service import SecurityService, CheckResult
from backend.logging_config import get_logger
from backend.exceptions import (
    LLMException,
    VectorStoreError,
    KnowledgeBaseEmptyError,
    NoRelevantDocumentsError,
)
from rag.prompts import DISCLAIMER_TEXT, QUESTION_TYPE_KEYWORDS
from backend.config import config
from backend.statistics import get_stats_instance

logger = get_logger(__name__)


def detect_question_type(question: str) -> Optional[str]:
    """检测问题类型

    Args:
        question: 用户问题

    Returns:
        问题类型：symptom/disease/medication/examination 或 None
    """
    question_lower = question.lower()

    # 统计每种类型的关键词匹配数
    type_scores = {}
    for qtype, keywords in QUESTION_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in question_lower)
        type_scores[qtype] = score

    # 返回得分最高的类型
    max_score = max(type_scores.values())
    if max_score > 0:
        for qtype, score in type_scores.items():
            if score == max_score:
                logger.info(f"检测到问题类型: {qtype} (匹配{score}个关键词)")
                return qtype

    return None


@dataclass
class QARequest:
    """问答请求"""
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None


@dataclass
class QAResponse:
    """问答响应"""
    answer: str
    sources: List[Dict[str, Any]]
    disclaimer: str
    is_safe: bool = True
    is_emergency: bool = False
    emergency_warning: Optional[str] = None
    question_type: Optional[str] = None
    confidence_level: str = "normal"


class QAService:
    """问答服务类 - 优化版"""

    def __init__(self, rag_engine: RAGEngine, security_service: SecurityService):
        self.rag_engine = rag_engine
        self.security_service = security_service

    def ask(self, request: QARequest) -> QAResponse:
        """处理问答请求 - 优化版

        Args:
            request: 问答请求

        Returns:
            QAResponse: 问答响应
        """
        # 0. 检测问题类型
        question_type = detect_question_type(request.question)
        logger.info(f"问题类型: {question_type or '通用'}, 问题: {request.question[:30]}...")

        # 1. 敏感词检查
        check_result = self.security_service.check_content(request.question)
        if not check_result.is_safe:
            logger.warning(f"检测到敏感内容，类别: {check_result.category}")
            return QAResponse(
                answer=check_result.warning_message,
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                is_safe=False,
                question_type=question_type
            )

        # 2. 紧急症状检查
        if self.security_service.is_emergency_symptom(request.question):
            logger.warning(f"检测到紧急症状: {request.question[:30]}...")
            return QAResponse(
                answer=self.security_service.get_emergency_message(),
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                is_safe=True,
                is_emergency=True,
                emergency_warning=self.security_service.get_emergency_message(),
                question_type=question_type
            )

        # 3. 检查知识库是否就绪
        if not self.rag_engine.is_ready():
            logger.info("知识库为空")
            return QAResponse(
                answer="知识库尚未就绪，请先在知识库管理页面上传医疗文档。",
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                question_type=question_type
            )

        # 4. 构建Prompt（含历史上下文）
        prompt = self.rag_engine.build_prompt(
            query=request.question,
            history=request.chat_history
        )

        # 5. 检索相关文档
        try:
            retrieved_docs = self.rag_engine.retrieve(request.question, top_k=config.TOP_K)
        except (VectorStoreError, EmbeddingError) as e:
            logger.error(f"检索失败: {e}")
            return QAResponse(
                answer=f"检索知识库时发生错误，请稍后重试。错误: {str(e)}",
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                question_type=question_type,
                confidence_level="low"
            )

        # 6. 检查检索结果的相关性
        if not retrieved_docs:
            logger.info("未找到相关文档")
            return QAResponse(
                answer="抱歉，在我的知识库中未找到与您问题相关的信息。建议您：\n1. 尝试使用不同的关键词\n2. 上传更多相关文档到知识库\n3. 咨询专业医生获取准确信息。",
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                question_type=question_type,
                confidence_level="low"
            )

        # 7. 置信度检查 - 优化版
        # 获取最高相似度分数
        top_score = retrieved_docs[0].get('score') if retrieved_docs else None
        confidence_level = "high"
        confidence_warning = ""

        if top_score is not None:
            # 转换为相似度百分比
            similarity = (1 - top_score) * 100
            if similarity < 60:
                confidence_level = "low"
                confidence_warning = f"⚠️ 知识库匹配度较低（{similarity:.0f}%），回答仅供参考"
            elif similarity < 75:
                confidence_level = "medium"
                confidence_warning = f"ℹ️ 知识库匹配度一般（{similarity:.0f}%）"

        # 8. 调用LLM生成回答（传入问题类型）
        try:
            answer = self.rag_engine.generate(prompt, retrieved_docs, question_type)
        except LLMException as e:
            logger.error(f"LLM生成失败: {e}")
            return QAResponse(
                answer=f"生成回答时发生错误，请稍后重试。错误: {str(e)}",
                sources=self.rag_engine.get_retrieved_sources(retrieved_docs),
                disclaimer=DISCLAIMER_TEXT,
                question_type=question_type,
                confidence_level="low"
            )
        except Exception as e:
            logger.error(f"生成回答时发生未知错误: {e}", exc_info=True)
            return QAResponse(
                answer="生成回答时发生未知错误，请稍后重试。",
                sources=self.rag_engine.get_retrieved_sources(retrieved_docs),
                disclaimer=DISCLAIMER_TEXT,
                question_type=question_type,
                confidence_level="low"
            )

        # 9. 如果相关性低，在回答前添加提示
        if confidence_warning:
            answer = f"{confidence_warning}\n\n{answer}"

        # 10. 提取来源信息
        sources = self.rag_engine.get_retrieved_sources(retrieved_docs)

        return QAResponse(
            answer=answer,
            sources=sources,
            disclaimer=DISCLAIMER_TEXT,
            question_type=question_type,
            confidence_level=confidence_level
        )

    def is_knowledge_base_ready(self) -> bool:
        """检查知识库是否就绪"""
        return self.rag_engine.is_ready()

    def get_document_count(self) -> int:
        """获取知识库中的文档数量"""
        return self.rag_engine.get_document_count()

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.rag_engine.get_cache_stats()

    def ask_stream(self, request: QARequest):
        """流式问答 - 返回生成器
        
        Args:
            request: 问答请求
            
        Yields:
            流式返回的回答内容
        """
        from typing import Generator
        import time
        
        # 获取统计实例
        stats = get_stats_instance()
        
        # 计时开始
        total_start = time.time()
        retrieval_start = time.time()
        
        # 0. 检测问题类型
        question_type = detect_question_type(request.question)
        logger.info(f"流式问答 - 问题类型: {question_type or '通用'}, 问题: {request.question[:30]}...")

        # 1. 敏感词检查
        check_result = self.security_service.check_content(request.question)
        if not check_result.is_safe:
            logger.warning(f"检测到敏感内容，类别: {check_result.category}")
            # 记录统计
            stats.record_question(
                question=request.question,
                question_type=question_type,
                success=False,
                has_result=False,
                response_time_ms=0,
                retrieval_time_ms=0,
                llm_time_ms=0,
                is_sensitive=True
            )
            yield check_result.warning_message
            return

        # 2. 紧急症状检查
        is_emergency = self.security_service.is_emergency_symptom(request.question)
        if is_emergency:
            logger.warning(f"检测到紧急症状: {request.question[:30]}...")
            yield self.security_service.get_emergency_message()
            return

        # 3. 检查知识库是否就绪
        if not self.rag_engine.is_ready():
            logger.info("知识库为空")
            yield "知识库尚未就绪，请先在知识库管理页面上传医疗文档。"
            return

        # 4. 构建Prompt（含历史上下文）
        prompt = self.rag_engine.build_prompt(
            query=request.question,
            history=request.chat_history
        )

        # 5. 检索相关文档
        retrieval_end = time.time()
        retrieval_time_ms = (retrieval_end - retrieval_start) * 1000
        
        try:
            retrieved_docs = self.rag_engine.retrieve(request.question, top_k=config.TOP_K)
        except (VectorStoreError, EmbeddingError) as e:
            logger.error(f"检索失败: {e}")
            # 记录失败统计
            stats.record_question(
                question=request.question,
                question_type=question_type,
                success=False,
                has_result=False,
                response_time_ms=(time.time() - total_start) * 1000,
                retrieval_time_ms=retrieval_time_ms,
                llm_time_ms=0,
                is_emergency=is_emergency
            )
            yield f"检索知识库时发生错误，请稍后重试。错误: {str(e)}"
            return

        # 6. 检查检索结果的相关性
        has_result = len(retrieved_docs) > 0
        if not retrieved_docs:
            logger.info("未找到相关文档")
            # 记录无结果统计
            stats.record_question(
                question=request.question,
                question_type=question_type,
                success=True,
                has_result=False,
                response_time_ms=(time.time() - total_start) * 1000,
                retrieval_time_ms=retrieval_time_ms,
                llm_time_ms=0,
                is_emergency=is_emergency
            )
            yield "抱歉，在我的知识库中未找到与您问题相关的信息。建议您：\n1. 尝试使用不同的关键词\n2. 上传更多相关文档到知识库\n3. 咨询专业医生获取准确信息。"
            return

        # 7. 置信度检查
        top_score = retrieved_docs[0].get('score') if retrieved_docs else None
        confidence_warning = ""
        
        if top_score is not None:
            similarity = (1 - top_score) * 100
            if similarity < 60:
                confidence_warning = f"⚠️ 知识库匹配度较低（{similarity:.0f}%），回答仅供参考\n\n"
            elif similarity < 75:
                confidence_warning = f"ℹ️ 知识库匹配度一般（{similarity:.0f}%）\n\n"

        # 7. 先输出置信度警告（如有）
        if confidence_warning:
            yield confidence_warning

        # 8. 流式调用LLM生成回答
        llm_start = time.time()
        try:
            for chunk in self.rag_engine.generate_stream(request.question, retrieved_docs, question_type, full_prompt=prompt):
                yield chunk
            # 流式完成后记录统计
            llm_end = time.time()
            llm_time_ms = (llm_end - llm_start) * 1000
            total_time_ms = (llm_end - total_start) * 1000
            stats.record_question(
                question=request.question,
                question_type=question_type,
                success=True,
                has_result=has_result,
                response_time_ms=total_time_ms,
                retrieval_time_ms=retrieval_time_ms,
                llm_time_ms=llm_time_ms,
                is_emergency=is_emergency
            )
        except LLMException as e:
            logger.error(f"LLM流式生成失败: {e}")
            yield f"\n\n生成回答时发生错误，请稍后重试。错误: {str(e)}"
            # 记录失败
            stats.record_question(
                question=request.question,
                question_type=question_type,
                success=False,
                has_result=has_result,
                response_time_ms=(time.time() - total_start) * 1000,
                retrieval_time_ms=retrieval_time_ms,
                llm_time_ms=0,
                is_emergency=is_emergency
            )
        except Exception as e:
            logger.error(f"流式生成回答时发生未知错误: {e}", exc_info=True)
            yield "\n\n生成回答时发生未知错误，请稍后重试。"
            # 记录失败
            stats.record_question(
                question=request.question,
                question_type=question_type,
                success=False,
                has_result=has_result,
                response_time_ms=(time.time() - total_start) * 1000,
                retrieval_time_ms=retrieval_time_ms,
                llm_time_ms=0,
                is_emergency=is_emergency
            )