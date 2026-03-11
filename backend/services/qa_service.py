"""问答服务模块 - 优化版，增加问题类型识别和统计"""

import time
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from collections import OrderedDict

from rag.core.engine import RAGEngine
from backend.services.security_service import SecurityService, CheckResult
from backend.services.question_type_detector import QuestionTypeDetector, get_question_type_detector
from backend.services.confidence_calculator import ConfidenceCalculator, get_confidence_calculator
from backend.logging_config import get_logger
from backend.exceptions import (
    LLMException,
    VectorStoreError,
    KnowledgeBaseEmptyError,
    NoRelevantDocumentsError,
)
from rag.core.prompts import DISCLAIMER_TEXT
from backend.config import config
from backend.statistics import get_stats_instance

logger = get_logger(__name__)


# 查询分析缓存（安全检查+问题类型检测结果）
_query_analysis_cache: OrderedDict = OrderedDict()


@dataclass
class QARequest:
    """问答请求"""
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None
    
    def __post_init__(self):
        """验证并限制 chat_history 长度"""
        if self.chat_history and len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]


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
    """问答服务类 - 优化版
    
    职责：
    - 协调各子服务处理问答请求
    - 管理问答流程
    """
    
    def __init__(
        self,
        rag_engine: RAGEngine,
        security_service: SecurityService,
        question_type_detector: QuestionTypeDetector = None,
        confidence_calculator: ConfidenceCalculator = None
    ):
        self.rag_engine = rag_engine
        self.security_service = security_service
        self.question_type_detector = question_type_detector or get_question_type_detector()
        self.confidence_calculator = confidence_calculator or get_confidence_calculator()
    
    def _check_safety(self, question: str) -> Optional[CheckResult]:
        """安全检查
        
        Returns:
            CheckResult if unsafe, None if safe
        """
        check_result = self.security_service.check_content(question)
        if not check_result.is_safe:
            logger.warning(f"检测到敏感内容，类别: {check_result.category}")
            return check_result
        return None
    
    def _check_emergency(self, question: str) -> bool:
        """紧急症状检查
        
        Returns:
            True if emergency symptom detected
        """
        return self.security_service.is_emergency_symptom(question)

    def _get_query_analysis_cache(self, question: str) -> Optional[Dict]:
        """获取查询分析缓存
        
        Args:
            question: 用户问题
            
        Returns:
            缓存的查询分析结果（安全检查+问题类型），None表示未命中
        """
        cache_key = hashlib.md5(question.encode('utf-8')).hexdigest()
        return _query_analysis_cache.get(cache_key)

    def _set_query_analysis_cache(self, question: str, question_type: str, is_safe: bool, is_emergency: bool):
        """设置查询分析缓存
        
        Args:
            question: 用户问题
            question_type: 问题类型
            is_safe: 是否安全
            is_emergency: 是否紧急
        """
        cache_key = hashlib.md5(question.encode('utf-8')).hexdigest()
        _query_analysis_cache[cache_key] = {
            "question_type": question_type,
            "is_safe": is_safe,
            "is_emergency": is_emergency
        }
        # LRU淘汰
        if len(_query_analysis_cache) > config.QUERY_CACHE_SIZE:
            _query_analysis_cache.popitem(last=False)

    def ask(self, request: QARequest) -> QAResponse:
        """处理问答请求 - 优化版
        
        流程：
        1. 检测问题类型
        2. 安全检查（敏感词+紧急症状）
        3. 检查知识库状态
        4. 检索文档
        5. 计算置信度
        6. 生成回答
        """
        # 0. 尝试从缓存获取查询分析结果
        cached = self._get_query_analysis_cache(request.question)
        
        if cached:
            # 使用缓存的问题类型
            question_type = cached["question_type"]
            logger.info(f"使用缓存的问题类型: {question_type or '通用'}, 问题: {request.question[:30]}...")
            
            # 如果缓存标记为不安全，直接返回警告
            if not cached["is_safe"]:
                check_result = self._check_safety(request.question)
                return QAResponse(
                    answer=check_result.warning_message if check_result else "内容不安全",
                    sources=[],
                    disclaimer=DISCLAIMER_TEXT,
                    is_safe=False,
                    question_type=question_type
                )
            
            # 如果缓存标记为紧急，直接返回警告
            if cached["is_emergency"]:
                logger.warning(f"检测到紧急症状(缓存): {request.question[:30]}...")
                return QAResponse(
                    answer=self.security_service.get_emergency_message(),
                    sources=[],
                    disclaimer=DISCLAIMER_TEXT,
                    is_safe=True,
                    is_emergency=True,
                    emergency_warning=self.security_service.get_emergency_message(),
                    question_type=question_type
                )
        else:
            # 0. 检测问题类型
            question_type = self.question_type_detector.detect(request.question)
            logger.info(f"问题类型: {question_type or '通用'}, 问题: {request.question[:30]}...")

        # 1. 敏感词检查
        check_result = self._check_safety(request.question)
        if check_result:
            return QAResponse(
                answer=check_result.warning_message,
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                is_safe=False,
                question_type=question_type
            )

        # 2. 紧急症状检查
        if self._check_emergency(request.question):
            logger.warning(f"检测到紧急症状: {request.question[:30]}...")
            # 缓存紧急状态
            if not cached:
                self._set_query_analysis_cache(request.question, question_type, True, True)
            return QAResponse(
                answer=self.security_service.get_emergency_message(),
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                is_safe=True,
                is_emergency=True,
                emergency_warning=self.security_service.get_emergency_message(),
                question_type=question_type
            )

        # 缓存安全的查询分析结果
        if not cached:
            self._set_query_analysis_cache(request.question, question_type, True, False)

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

        # 7. 计算置信度
        confidence_result = self.confidence_calculator.calculate_with_sources(retrieved_docs)
        confidence_level = confidence_result["confidence_level"]
        confidence_warning = confidence_result["warning"]

        # 8. 调用LLM生成回答（传入问题类型）
        try:
            answer = self.rag_engine.generate(prompt, retrieved_docs, question_type)
        except LLMException as e:
            logger.error(f"LLM生成失败: {e}")
            return QAResponse(
                answer=f"生成回答时发生错误，请稍后重试。错误: {str(e)}",
                sources=confidence_result.get("sources", []),
                disclaimer=DISCLAIMER_TEXT,
                question_type=question_type,
                confidence_level="low"
            )
        except Exception as e:
            logger.error(f"生成回答时发生未知错误: {e}", exc_info=True)
            return QAResponse(
                answer="生成回答时发生未知错误，请稍后重试。",
                sources=confidence_result.get("sources", []),
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

    # =========================================================================
    # 公共方法（供内部调用和ask/ask_stream共用）
    # =========================================================================

    def _analyze_query(self, question: str) -> Dict[str, Any]:
        """分析查询：问题类型检测 + 安全检查 + 紧急症状检查

        Args:
            question: 用户问题

        Returns:
            包含以下键的字典：
            - question_type: 问题类型
            - is_safe: 是否安全
            - is_emergency: 是否紧急
            - check_result: 安全检查结果（如果不安全）
            - emergency_message: 紧急消息（如果紧急）
        """
        # 尝试从缓存获取
        cached = self._get_query_analysis_cache(question)

        if cached:
            logger.info(f"使用缓存的分析结果: {cached.get('question_type') or '通用'}, 问题: {question[:30]}...")
            return {
                "question_type": cached["question_type"],
                "is_safe": cached["is_safe"],
                "is_emergency": cached["is_emergency"],
                "check_result": None if cached["is_safe"] else self._check_safety(question),
                "emergency_message": self.security_service.get_emergency_message() if cached["is_emergency"] else None,
                "from_cache": True
            }

        # 检测问题类型
        question_type = self.question_type_detector.detect(question)
        logger.info(f"问题类型: {question_type or '通用'}, 问题: {question[:30]}...")

        # 安全检查
        check_result = self._check_safety(question)
        is_safe = check_result is None

        # 紧急症状检查
        is_emergency = self._check_emergency(question)

        # 缓存结果
        self._set_query_analysis_cache(question, question_type, is_safe, is_emergency)

        return {
            "question_type": question_type,
            "is_safe": is_safe,
            "is_emergency": is_emergency,
            "check_result": check_result,
            "emergency_message": self.security_service.get_emergency_message() if is_emergency else None,
            "from_cache": False
        }

    def _build_response(
        self,
        question: str,
        question_type: Optional[str],
        is_safe: bool,
        is_emergency: bool,
        check_result: Optional[CheckResult],
        emergency_message: Optional[str],
        retrieved_docs: Optional[List[Any]],
        confidence_result: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None
    ) -> QAResponse:
        """构建问答响应（统一响应生成逻辑）

        Args:
            question: 用户问题
            question_type: 问题类型
            is_safe: 是否安全
            is_emergency: 是否紧急
            check_result: 安全检查结果
            emergency_message: 紧急消息
            retrieved_docs: 检索到的文档
            confidence_result: 置信度计算结果
            answer: 已生成的回答（可选）

        Returns:
            QAResponse对象
        """
        # 不安全内容
        if not is_safe:
            return QAResponse(
                answer=check_result.warning_message if check_result else "内容不安全",
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                is_safe=False,
                question_type=question_type
            )

        # 紧急症状
        if is_emergency:
            return QAResponse(
                answer=emergency_message or self.security_service.get_emergency_message(),
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                is_safe=True,
                is_emergency=True,
                emergency_warning=emergency_message or self.security_service.get_emergency_message(),
                question_type=question_type
            )

        # 知识库为空
        if retrieved_docs is None:
            return QAResponse(
                answer="知识库尚未就绪，请先在知识库管理页面上传医疗文档。",
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                question_type=question_type
            )

        # 无检索结果
        if not retrieved_docs:
            return QAResponse(
                answer="抱歉，在我的知识库中未找到与您问题相关的信息。建议您：\n1. 尝试使用不同的关键词\n2. 上传更多相关文档到知识库\n3. 咨询专业医生获取准确信息。",
                sources=[],
                disclaimer=DISCLAIMER_TEXT,
                question_type=question_type,
                confidence_level="low"
            )

        # 正常回答
        confidence_level = confidence_result.get("confidence_level", "normal") if confidence_result else "normal"
        confidence_warning = confidence_result.get("warning", "") if confidence_result else ""

        # 添加置信度警告
        if confidence_warning and answer:
            answer = f"{confidence_warning}\n\n{answer}"

        # 提取来源
        sources = self.rag_engine.get_retrieved_sources(retrieved_docs)

        return QAResponse(
            answer=answer or "",
            sources=sources,
            disclaimer=DISCLAIMER_TEXT,
            question_type=question_type,
            confidence_level=confidence_level
        )

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

        # 0. 使用公共方法分析查询
        analysis = self._analyze_query(request.question)
        question_type = analysis["question_type"]
        is_safe = analysis["is_safe"]
        is_emergency = analysis["is_emergency"]
        check_result = analysis["check_result"]
        emergency_message = analysis["emergency_message"]

        # 1. 不安全内容处理
        if not is_safe:
            logger.warning(f"检测到敏感内容，类别: {check_result.category if check_result else '未知'}")
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
            yield check_result.warning_message if check_result else "内容不安全"
            return

        # 2. 紧急症状处理
        if is_emergency:
            logger.warning(f"检测到紧急症状: {request.question[:30]}...")
            stats.record_question(
                question=request.question,
                question_type=question_type,
                success=True,
                has_result=False,
                response_time_ms=(time.time() - total_start) * 1000,
                retrieval_time_ms=0,
                llm_time_ms=0,
                is_emergency=True
            )
            yield emergency_message or self.security_service.get_emergency_message()
            return

        # 3. 检查知识库是否就绪
        if not self.rag_engine.is_ready():
            logger.info("知识库为空")
            stats.record_question(
                question=request.question,
                question_type=question_type,
                success=True,
                has_result=False,
                response_time_ms=(time.time() - total_start) * 1000,
                retrieval_time_ms=0,
                llm_time_ms=0,
                is_emergency=False
            )
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

        # 7. 置信度检查 - 使用置信度计算器
        confidence_level, confidence_warning = self.confidence_calculator.calculate(retrieved_docs)

        if confidence_warning:
            confidence_warning = confidence_warning + "\n\n"

        # 先输出置信度警告（如有）
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
            # 记录缓存统计
            cache_stats = self.rag_engine.get_cache_stats()
            if cache_stats.get("llm_response_cache"):
                stats.record_cache_stats(
                    hits=cache_stats["llm_response_cache"].get("hits", 0),
                    misses=cache_stats["llm_response_cache"].get("misses", 0)
                )
        except LLMException as e:
            logger.error(f"LLM流式生成失败: {e}")
            yield f"\n\n生成回答时发生错误，请稍后重试。错误: {str(e)}"
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