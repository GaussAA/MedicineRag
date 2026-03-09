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
    """简单的内存限流器"""
    
    def __init__(self):
        self._requests = defaultdict(list)
        self._lock = threading.Lock()
    
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
            return True
    
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
    # 生产环境：严格要求配置，不使用通配符
    if cors_env and cors_env != "*":
        cors_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    else:
        # 未配置时只允许本应用域名（防止意外开放）
        cors_origins = []
        logger.warning("生产环境未配置CORS_ALLOWED_ORIGINS，已禁用跨域请求！")
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


# 限流中间件
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
