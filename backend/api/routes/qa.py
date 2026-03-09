"""问答API路由"""

from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.models import (
    QARequest, QAResponse, StreamQARequest,
    ErrorResponse
)
from backend.services.qa_service import QAService, QARequest as QARequestModel
from backend.services.security_service import SecurityService
from rag.engine import RAGEngine
from backend.logging_config import get_logger
from backend.exceptions import LLMException, VectorStoreError

logger = get_logger(__name__)

# 创建路由
router = APIRouter(prefix="/qa", tags=["问答"])


@lru_cache()
def get_rag_engine() -> RAGEngine:
    """获取RAG引擎实例（单例）"""
    return RAGEngine()


@lru_cache()
def get_security_service() -> SecurityService:
    """获取安全服务实例（单例）"""
    return SecurityService()


def get_qa_service(
    rag_engine: RAGEngine = Depends(get_rag_engine),
    security_service: SecurityService = Depends(get_security_service)
) -> QAService:
    """获取QA服务实例（依赖注入）"""
    return QAService(rag_engine, security_service)


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
                
                # 发送心跳（保持连接）
                import time
                heartbeat_count = 0
                
                for chunk in qa_service.ask_stream(qa_request):
                    # 转义换行符避免SSE解析问题
                    escaped_chunk = chunk.replace('\n', '\\n')
                    yield f"data: {{\"type\": \"content\", \"content\": \"{escaped_chunk}\"}}\n\n"
                    
                    # 每30秒发送一次心跳
                    heartbeat_count += 1
                    if heartbeat_count % 30 == 0:
                        yield "data: {\"type\": \"heartbeat\"}\n\n"
                
                # 发送完成标志
                yield "data: {\"type\": \"done\"}\n\n"
                
            except Exception as e:
                logger.error(f"流式问答错误: {e}")
                import json
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
