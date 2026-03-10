"""FastAPI应用入口"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import datetime, timedelta
import threading

from backend.logging_config import get_logger
from backend.config import config

logger = get_logger(__name__)


class RateLimiter:
    """简单的内存限流器（优化版）"""
    
    def __init__(self, cleanup_threshold: int = 1000):
        self._requests = defaultdict(list)
        self._lock = threading.Lock()
        self._cleanup_counter = 0
        self._cleanup_threshold = cleanup_threshold  # 每N次请求清理一次
    
    def is_allowed(self, client_id: str, max_requests: int, window_seconds: int) -> bool:
        """检查请求是否允许
        
        Args:
            client_id: 客户端标识
            max_requests: 窗口期内最大请求数
            window_seconds: 时间窗口（秒）
            
        Returns:
            是否允许请求
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)
        
        with self._lock:
            # 清理过期请求记录
            self._requests[client_id] = [
                t for t in self._requests[client_id] 
                if t > window_start
            ]
            
            # 检查是否超限
            if len(self._requests[client_id]) >= max_requests:
                return False
            
            # 记录新请求
            self._requests[client_id].append(now)
            
            # 定期清理非活跃客户端（防止内存泄漏）
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_threshold:
                self._cleanup_inactive_clients(window_start)
                self._cleanup_counter = 0
            
            return True
    
    def _cleanup_inactive_clients(self, window_start: datetime):
        """清理非活跃客户端"""
        inactive = [
            client for client, times in self._requests.items()
            if not times or all(t <= window_start for t in times)
        ]
        for client in inactive:
            del self._requests[client]
    
    def get_remaining(self, client_id: str, max_requests: int, window_seconds: int) -> int:
        """获取剩余请求数"""
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)
        
        with self._lock:
            self._requests[client_id] = [
                t for t in self._requests[client_id] 
                if t > window_start
            ]
            return max(0, max_requests - len(self._requests[client_id]))


# 全局限流器
rate_limiter = RateLimiter()


def get_client_id(request: Request) -> str:
    """获取客户端标识（优先X-Forwarded-For）"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_ollama_connection() -> dict:
    """检查Ollama连接"""
    try:
        from ollama import Client
        client = Client(host="http://localhost:11434")
        # 尝试获取模型列表
        client.list()
        return {"status": "healthy", "message": "Ollama连接正常"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Ollama连接失败: {str(e)[:50]}"}


def check_chroma_connection() -> dict:
    """检查Chroma连接"""
    try:
        import chromadb
        from backend.config import config
        client = chromadb.PersistentClient(path=str(config.get_chroma_dir()))
        # 尝试获取collection
        client.get_or_create_collection("test_connection")
        return {"status": "healthy", "message": "Chroma连接正常"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Chroma连接失败: {str(e)[:50]}"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("API服务启动...")
    
    # 预热检查
    logger.info("检查服务连接...")
    ollama_status = check_ollama_connection()
    chroma_status = check_chroma_connection()
    logger.info(f"Ollama状态: {ollama_status['status']}")
    logger.info(f"Chroma状态: {chroma_status['status']}")
    
    yield
    
    # 关闭时 - 刷新统计缓存
    logger.info("API服务关闭...")
    try:
        from backend.statistics import get_stats_instance
        stats = get_stats_instance()
        stats.flush()
        logger.info("统计缓存已刷新")
    except Exception as e:
        logger.warning(f"刷新统计缓存失败: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="医疗知识问答系统API",
    description="基于RAG技术的医疗知识问答系统的REST API接口",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
# 从环境变量读取允许的来源，生产环境应限制具体域名
import os
cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()

# 安全策略：默认根据环境决定
is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"

if is_production:
    # 生产环境：严格要求配置
    if cors_env and cors_env != "*":
        cors_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    else:
        # 未配置时默认允许本地服务（避免完全无法访问）
        cors_origins = ["http://localhost", "http://127.0.0.1"]
        logger.warning("生产环境未配置CORS_ALLOWED_ORIGINS，默认仅允许本机访问")
else:
    # 开发环境：灵活配置
    if cors_env:
        cors_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
        if "*" in cors_origins:
            logger.warning("开发环境使用了通配符CORS配置，生产环境建议指定具体域名")
    else:
        # 默认允许本地开发
        cors_origins = ["http://localhost:8501", "http://localhost:3000", "http://127.0.0.1:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 安全响应头中间件
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    # 防止内容类型被猜测
    response.headers["X-Content-Type-Options"] = "nosniff"
    # 防止点击劫持
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    # XSS 保护
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # 引用策略
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ============================================================================
# 全局异常处理器
# ============================================================================

from fastapi import HTTPException
from backend.exceptions import (
    MedicalRAGException,
    LLMException,
    VectorStoreError,
    DocumentParseError,
    SecurityException,
    ServiceException,
    RAGEngineException
)


# 定义别名，使代码更清晰
RateLimitException = ServiceException  # 使用ServiceException作为限流异常
MedicalQAException = MedicalRAGException  # 使用基础异常类


@app.exception_handler(MedicalQAException)
async def medical_qa_exception_handler(request: Request, exc: MedicalQAException):
    """医疗问答系统基础异常处理器"""
    logger.error(f"医疗问答系统异常: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code or 500,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details
        }
    )


@app.exception_handler(LLMException)
async def llm_exception_handler(request: Request, exc: LLMException):
    """LLM异常处理器"""
    logger.error(f"LLM错误: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "error": "LLM服务暂时不可用",
            "error_code": "LLM_ERROR",
            "details": {"message": exc.message}
        }
    )


@app.exception_handler(VectorStoreError)
async def vector_store_exception_handler(request: Request, exc: VectorStoreError):
    """向量存储异常处理器"""
    logger.error(f"向量存储错误: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "error": "向量数据库服务暂时不可用",
            "error_code": "VECTOR_STORE_ERROR",
            "details": {"message": exc.message}
        }
    )


@app.exception_handler(DocumentParseError)
async def document_parse_exception_handler(request: Request, exc: DocumentParseError):
    """文档解析异常处理器"""
    logger.error(f"文档解析错误: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=422,
        content={
            "error": "文档解析失败",
            "error_code": "DOCUMENT_PARSE_ERROR",
            "details": {"message": exc.message}
        }
    )


@app.exception_handler(SecurityException)
async def security_exception_handler(request: Request, exc: SecurityException):
    """安全异常处理器"""
    logger.warning(f"安全检查失败: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "error_code": exc.error_code or "SECURITY_ERROR",
            "details": exc.details
        }
    )


@app.exception_handler(RateLimitException)
async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
    """限流异常处理器"""
    logger.warning(f"限流触发: {exc.message}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "请求过于频繁",
            "error_code": "RATE_LIMIT_ERROR",
            "details": {"retry_after": exc.retry_after}
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """FastAPI HTTPException处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}"
        }
    )


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    """值错误异常处理器"""
    logger.error(f"值错误: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=400,
        content={
            "error": "请求参数无效",
            "error_code": "VALUE_ERROR",
            "details": {"message": str(exc)}
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器（最后一道防线）"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误",
            "error_code": "INTERNAL_ERROR",
            "details": {"message": "请联系管理员"}
        }
    )


# ============================================================================
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """API限流中间件（配置化版本）
    
    - 问答接口: 可通过 RATE_LIMIT_QA_MAX 配置
    - 文档上传: 可通过 RATE_LIMIT_UPLOAD_MAX 配置
    - 其他接口: 可通过 RATE_LIMIT_OTHER_MAX 配置
    - 时间窗口: 可通过 RATE_LIMIT_WINDOW 配置（秒）
    """
    # 获取限流配置
    rate_limit_enabled = config.ENABLE_RATE_LIMIT
    
    if rate_limit_enabled:
        client_id = get_client_id(request)
        path = request.url.path
        
        # 根据接口类型设置不同限流规则（使用配置值）
        if "/qa/" in path:
            max_requests = config.RATE_LIMIT_QA_MAX
        elif "/docs/upload" in path:
            max_requests = config.RATE_LIMIT_UPLOAD_MAX
        else:
            max_requests = config.RATE_LIMIT_OTHER_MAX
        
        window_seconds = config.RATE_LIMIT_WINDOW
        
        # 检查是否允许请求
        if not rate_limiter.is_allowed(client_id, max_requests, window_seconds):
            remaining = 0
        else:
            remaining = rate_limiter.get_remaining(client_id, max_requests, window_seconds)
        
        # 添加响应头
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        # 如果超过限流，返回429
        if remaining == 0:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "请求过于频繁，请稍后再试",
                    "retry_after": window_seconds
                }
            )
        
        return response
    
    return await call_next(request)


# 注册路由
from backend.api.routes import qa, docs

app.include_router(qa.router, prefix="/api")
app.include_router(docs.router, prefix="/api")


# 根路径
@app.get("/")
async def root():
    return {
        "name": "医疗知识问答系统API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


# 基础健康检查
@app.get("/health")
async def health():
    return {"status": "healthy"}


# 详细健康检查
@app.get("/health/detailed")
async def detailed_health():
    """详细健康检查 - 检查所有依赖服务"""
    ollama_status = check_ollama_connection()
    chroma_status = check_chroma_connection()
    
    # 检查知识库状态
    kb_status = {"status": "unknown", "document_count": 0}
    try:
        from rag.engine import RAGEngine
        engine = RAGEngine()
        kb_status["document_count"] = engine.get_document_count()
        kb_status["status"] = "ready" if kb_status["document_count"] > 0 else "empty"
    except Exception as e:
        kb_status["status"] = "error"
        kb_status["error"] = str(e)[:100]
    
    overall = "healthy" if (
        ollama_status["status"] == "healthy" and 
        chroma_status["status"] == "healthy"
    ) else "degraded"
    
    return {
        "status": overall,
        "ollama": ollama_status,
        "chroma": chroma_status,
        "knowledge_base": kb_status
    }


# 性能指标端点
@app.get("/metrics")
async def metrics():
    """系统性能指标"""
    try:
        stats = get_stats_instance().get_summary()
        
        return {
            "total_questions": stats.get("total_questions", 0),
            "successful_answers": stats.get("successful_answers", 0),
            "failed_answers": stats.get("failed_answers", 0),
            "success_rate": stats.get("success_rate", "0%"),
            "avg_response_time_ms": round(stats.get("avg_response_time_ms", 0), 2),
            "avg_retrieval_time_ms": round(stats.get("avg_retrieval_time_ms", 0), 2),
            "avg_llm_time_ms": round(stats.get("avg_llm_time_ms", 0), 2),
            "cache_hits": stats.get("cache_hits", 0),
            "cache_misses": stats.get("cache_misses", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", "0%"),
            "sensitive_blocked": stats.get("sensitive_blocked", 0),
            "emergency_warnings": stats.get("emergency_warnings", 0),
        }
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
