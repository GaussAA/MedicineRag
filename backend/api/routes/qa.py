"""问答API路由"""

import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.models import (
    QARequest, QAResponse, StreamQARequest,
    ErrorResponse
)
from backend.api.dependencies import (
    get_qa_service_dep as get_qa_service,
    get_medical_agent_dep as get_medical_agent
)
from backend.services.qa_service import QAService, QARequest as QARequestModel
from backend.logging_config import get_logger
from backend.exceptions import LLMException, VectorStoreError, EmbeddingError
from rag.agents.medical_agent import MedicalAgent

logger = get_logger(__name__)

# 创建路由
router = APIRouter(prefix="/qa", tags=["问答"])


@router.post("/ask", response_model=QAResponse)
async def ask(request: QARequest, qa_service: QAService = Depends(get_qa_service)):
    """问答接口（非流式）
    
    Args:
        request: 问答请求
        
    Returns:
        QAResponse: 问答响应
    """
    try:
        # 构建请求
        qa_request = QARequestModel(
            question=request.question,
            chat_history=request.history
        )
        
        # 调用服务
        response = qa_service.ask(qa_request)
        
        return QAResponse(
            answer=response.answer,
            sources=response.sources,
            disclaimer=response.disclaimer,
            question_type=response.question_type,
            confidence_level=response.confidence_level
        )
        
    except VectorStoreError as e:
        logger.error(f"向量存储错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except LLMException as e:
        logger.error(f"LLM错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"问答错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_qa(request: StreamQARequest, qa_service: QAService = Depends(get_qa_service)):
    """流式问答接口
    
    Args:
        request: 问答请求
        
    Returns:
        StreamingResponse: 流式响应
    """
    try:
        # 构建请求
        qa_request = QARequestModel(
            question=request.question,
            chat_history=request.history
        )
        
        def generate():
            """生成器函数 - 改进版"""
            try:
                # 发送开始标志
                yield "data: {\"type\": \"start\"}\n\n"
                
                # 预检索sources（在流式生成前获取）
                try:
                    retrieved_docs = qa_service.rag_engine.retrieve(request.question, top_k=5)
                    if retrieved_docs:
                        sources = qa_service.rag_engine.get_retrieved_sources(retrieved_docs)
                        # 先发送sources
                        for src in sources:
                            src_data = json.dumps({"type": "source", "data": src})
                            yield f"data: {src_data}\n\n"
                except Exception as src_err:
                    logger.warning(f"预检索sources失败: {src_err}")
                
                # 发送心跳（保持连接）
                heartbeat_count = 0
                
                for chunk in qa_service.ask_stream(qa_request):
                    # 使用json.dumps正确转义所有特殊字符
                    escaped_chunk = json.dumps(chunk, ensure_ascii=False)[1:-1]  # 去掉首尾引号
                    yield f"data: {{\"type\": \"content\", \"content\": \"{escaped_chunk}\"}}\n\n"
                    
                    # 心跳优化：每60秒发送一次（减少网络开销）
                    heartbeat_count += 1
                    if heartbeat_count % 60 == 0:
                        yield "data: {\"type\": \"heartbeat\"}\n\n"
                
                # 发送完成标志
                yield "data: {\"type\": \"done\"}\n\n"
                
            except Exception as e:
                logger.error(f"流式问答错误: {e}")
                error_data = json.dumps({
                    "type": "error",
                    "message": str(e)[:200],
                    "hint": "请刷新页面重试，如果问题持续存在请联系管理员"
                })
                yield f"data: {error_data}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"流式问答错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "qa"}


# ============================================================================
# Agent API
# ============================================================================

from pydantic import BaseModel


class AgentQARequest(BaseModel):
    """Agent 问答请求"""
    question: str
    history: Optional[list] = []
    session_id: Optional[str] = None
    enable_followup: bool = True
    enable_knowledge_gap: bool = True


class AgentQAResponse(BaseModel):
    """Agent 问答响应"""
    answer: str
    sources: list
    confidence: float
    disclaimer: str
    question_type: str = "unknown"
    followup_questions: list = []
    knowledge_gaps: list = []
    requires_followup: bool = False
    steps: list = []


@router.post("/agent", response_model=AgentQAResponse)
async def agent_qa(
    request: AgentQARequest,
    agent: MedicalAgent = Depends(get_medical_agent)
):
    """Agent 问答接口
    
    使用 Agent 模式进行问答，支持：
    - ReAct 推理
    - 工具调用（检索、安全检查、追问等）
    - 知识缺口识别
    - 主动追问
    
    Args:
        request: Agent 问答请求
        
    Returns:
        AgentQAResponse: Agent 问答响应
    """
    try:
        # 生成 session_id
        session_id = request.session_id or str(uuid.uuid4())
        
        # 构建上下文
        context = {
            "session_id": session_id,
            "history": request.history,
            "enable_followup": request.enable_followup,
            "enable_knowledge_gap": request.enable_knowledge_gap
        }
        
        # 调用 Agent
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: asyncio.run(agent.execute(request.question, context))
        )
        
        return AgentQAResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            disclaimer=result.disclaimer,
            followup_questions=result.followup_questions,
            knowledge_gaps=result.knowledge_gaps,
            requires_followup=result.requires_followup,
            steps=[{
                "step_num": s.step_num,
                "thought": s.thought,
                "action": s.action,
                "observation": s.observation,
                "reflection": s.reflection
            } for s in result.steps]
        )
        
    except Exception as e:
        logger.error(f"Agent 问答错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/stream")
async def stream_agent_qa(
    request: AgentQARequest,
    agent: MedicalAgent = Depends(get_medical_agent)
):
    """流式 Agent 问答接口
    
    Args:
        request: Agent 问答请求
        
    Returns:
        StreamingResponse: 流式响应
    """
    try:
        # 生成 session_id
        session_id = request.session_id or str(uuid.uuid4())
        
        # 构建上下文
        context = {
            "session_id": session_id,
            "history": request.history,
            "enable_followup": request.enable_followup,
            "enable_knowledge_gap": request.enable_knowledge_gap
        }
        
        def generate():
            """生成器函数"""
            try:
                # 发送开始标志
                yield "data: {\"type\": \"start\"}\n\n"
                
                # 使用线程执行异步 Agent
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # 获取异步结果
                    result = loop.run_until_complete(
                        agent.execute(request.question, context)
                    )
                    
                    # 发送推理步骤
                    if result.steps:
                        steps_data = json.dumps({
                            "type": "steps",
                            "data": [{
                                "step_num": s.step_num,
                                "thought": s.thought,
                                "action": s.action,
                                "observation": s.observation
                            } for s in result.steps]
                        }, ensure_ascii=False)
                        yield f"data: {steps_data}\n\n"
                    
                    # 发送来源
                    for src in result.sources:
                        src_data = json.dumps({"type": "source", "data": src})
                        yield f"data: {src_data}\n\n"
                    
                    # 发送回答（流式）
                    words = result.answer.split()
                    for i, word in enumerate(words):
                        content_data = json.dumps({
                            "type": "content",
                            "content": word + (" " if i < len(words) - 1 else "")
                        }, ensure_ascii=False)
                        yield f"data: {content_data}\n\n"
                        
                        # 每10个字发送心跳
                        if i > 0 and i % 10 == 0:
                            yield "data: {\"type\": \"heartbeat\"}\n\n"
                    
                    # 发送追问
                    if result.followup_questions:
                        followup_data = json.dumps({
                            "type": "followup",
                            "data": result.followup_questions,
                            "requires_followup": result.requires_followup
                        }, ensure_ascii=False)
                        yield f"data: {followup_data}\n\n"
                    
                    # 发送知识缺口
                    if result.knowledge_gaps:
                        gap_data = json.dumps({
                            "type": "knowledge_gaps",
                            "data": result.knowledge_gaps
                        }, ensure_ascii=False)
                        yield f"data: {gap_data}\n\n"
                    
                    # 发送置信度
                    conf_data = json.dumps({
                        "type": "confidence",
                        "data": result.confidence
                    })
                    yield f"data: {conf_data}\n\n"
                    
                finally:
                    loop.close()
                
                # 发送完成标志
                yield "data: {\"type\": \"done\"}\n\n"
                
            except Exception as e:
                logger.error(f"流式Agent问答错误: {e}")
                error_data = json.dumps({
                    "type": "error",
                    "message": str(e)[:200],
                    "hint": "请刷新页面重试"
                })
                yield f"data: {error_data}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"流式Agent问答错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
