"""文档管理API路由"""

from functools import lru_cache
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from backend.api.models import (
    UploadResponse, DocListResponse, DocInfo,
    DeleteResponse, RebuildResponse, StatsResponse,
    ErrorResponse
)
from backend.services.doc_service import DocService
from rag.engine import RAGEngine
from backend.logging_config import get_logger

logger = get_logger(__name__)

# 创建路由
router = APIRouter(prefix="/docs", tags=["文档管理"])


@lru_cache()
def get_rag_engine() -> RAGEngine:
    """获取RAG引擎实例（单例）"""
    return RAGEngine()


def get_doc_service(rag_engine: RAGEngine = Depends(get_rag_engine)) -> DocService:
    """获取文档服务实例（依赖注入）"""
    return DocService(rag_engine)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), doc_service: DocService = Depends(get_doc_service)):
    """上传文档接口
    
    Args:
        file: 上传的文件
        
    Returns:
        UploadResponse: 上传结果
    """
    try:
        # 保存上传的文件
        import tempfile
        import os
        
        # 创建临时文件
        suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # 上传并索引
            result = doc_service.upload_document(tmp_path, file.filename)
            
            return UploadResponse(
                status=result.get("status", "error"),
                message=result.get("message", ""),
                file_name=file.filename,
                doc_count=result.get("doc_count")
            )
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"上传文档错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=DocListResponse)
async def list_documents(doc_service: DocService = Depends(get_doc_service)):
    """获取文档列表
    
    Returns:
        DocListResponse: 文档列表
    """
    try:
        docs = doc_service.list_documents()
        
        doc_list = [
            DocInfo(
                doc_id=doc.get("id", ""),
                file_name=doc.get("file_name", ""),
                chunk_count=doc.get("chunk_count", 0),
                size=doc.get("size", 0)
            )
            for doc in docs
        ]
        
        return DocListResponse(documents=doc_list)
        
    except Exception as e:
        logger.error(f"获取文档列表错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str, doc_service: DocService = Depends(get_doc_service)):
    """删除文档
    
    Args:
        doc_id: 文档ID
        
    Returns:
        DeleteResponse: 删除结果
    """
    try:
        result = doc_service.delete_document(doc_id)
        
        return DeleteResponse(
            status=result.get("status", "success"),
            message=result.get("message", "")
        )
        
    except Exception as e:
        logger.error(f"删除文档错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_index(doc_service: DocService = Depends(get_doc_service)):
    """重建索引
    
    Returns:
        RebuildResponse: 重建结果
    """
    try:
        result = doc_service.rebuild_index()
        
        return RebuildResponse(
            status=result.get("status", "success"),
            message=result.get("message", ""),
            doc_count=result.get("doc_count")
        )
        
    except Exception as e:
        logger.error(f"重建索引错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear", response_model=DeleteResponse)
async def clear_knowledge_base(doc_service: DocService = Depends(get_doc_service)):
    """清空知识库
    
    Returns:
        DeleteResponse: 清空结果
    """
    try:
        result = doc_service.clear_knowledge_base()
        
        return DeleteResponse(
            status=result.get("status", "success"),
            message=result.get("message", "")
        )
        
    except Exception as e:
        logger.error(f"清空知识库错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats(doc_service: DocService = Depends(get_doc_service)):
    """获取统计信息
    
    Returns:
        StatsResponse: 统计信息
    """
    try:
        stats = doc_service.get_stats()
        
        return StatsResponse(
            total_questions=0,  # API层面不统计问答
            successful_answers=0,
            success_rate="N/A",
            avg_response_time_ms=0,
            question_types={}
        )
        
    except Exception as e:
        logger.error(f"获取统计错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
