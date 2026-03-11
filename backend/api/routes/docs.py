"""文档管理API路由"""

from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import threading
from concurrent.futures import ThreadPoolExecutor

from backend.api.models import (
    UploadResponse, DocListResponse, DocInfo,
    DeleteResponse, RebuildResponse, StatsResponse,
    ErrorResponse
)
from backend.api.dependencies import get_doc_service_dep as get_doc_service
from backend.services.doc_service import DocService
from backend.statistics import get_stats_instance
from backend.logging_config import get_logger

logger = get_logger(__name__)

# 创建路由
router = APIRouter(prefix="/docs", tags=["文档管理"])

# 创建线程池用于后台文档处理
_executor = ThreadPoolExecutor(max_workers=4)


def _process_document_task(doc_service: DocService, tmp_path: str, file_name: str):
    """后台文档处理任务"""
    try:
        logger.info(f"开始后台处理文档: {file_name}")
        logger.info(f"DEBUG: 调用 doc_service.upload_document, tmp_path={tmp_path}")
        result = doc_service.upload_document(tmp_path, file_name)
        logger.info(f"文档处理完成: {file_name}, 结果: {result.get('status')}, 详情: {result.get('message', '')}")
    except Exception as e:
        logger.error(f"后台文档处理失败: {e}", exc_info=True)
    finally:
        # 后台任务完成后清理临时文件
        try:
            import os
            os.unlink(tmp_path)
            logger.info(f"临时文件已清理: {tmp_path}")
        except OSError:
            pass


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_service: DocService = Depends(get_doc_service)
):
    """上传文档接口（异步后台处理）
    
    Args:
        file: 上传的文件
        
    Returns:
        UploadResponse: 上传结果（立即返回，文档处理在后台进行）
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
        
        # 检查文件大小（调试用）
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"上传文件大小: {file_size_mb:.2f}MB")
        
        # 使用线程池异步处理
        _executor.submit(_process_document_task, doc_service, tmp_path, file.filename)
        
        return UploadResponse(
            status="processing",
            message=f"文件正在后台处理中...",
            file_name=file.filename,
            doc_count=None
        )
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
                doc_id=doc.get("name", ""),  # 使用 name 作为 doc_id
                file_name=doc.get("name", ""),
                chunk_count=doc.get("chunk_count", 0),
                size=doc.get("size", 0),  # 原始字节大小
                size_formatted=doc.get("size_formatted", "")  # 格式化大小
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
    """重建索引（异步后台处理）
    
    Returns:
        RebuildResponse: 重建结果
    """
    try:
        result = doc_service.rebuild_index(async_mode=True)
        
        return RebuildResponse(
            status=result.get("status", "success"),
            message=result.get("message", ""),
            doc_count=result.get("doc_count")
        )
        
    except Exception as e:
        logger.error(f"重建索引错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rebuild/status")
async def get_rebuild_status(doc_service: DocService = Depends(get_doc_service)):
    """获取重建索引任务状态
    
    Returns:
        重建任务状态
    """
    try:
        status = doc_service.get_rebuild_status()
        return status
    except Exception as e:
        logger.error(f"获取重建状态错误: {e}", exc_info=True)
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
    """获取统计信息（知识库 + 问答）
    
    Returns:
        StatsResponse: 统计信息
    """
    try:
        # 获取知识库统计
        doc_stats = doc_service.get_stats()
        
        # 获取问答统计
        stats_instance = get_stats_instance()
        qa_stats = stats_instance.get_summary()
        
        return StatsResponse(
            # 知识库统计
            document_count=doc_stats.get("document_count", 0),
            indexed_chunks=doc_stats.get("indexed_chunks", 0),
            total_size=doc_stats.get("total_size", "0 KB"),
            # 问答统计
            total_questions=qa_stats.get("total_questions", 0),
            successful_answers=qa_stats.get("successful_answers", 0),
            success_rate=qa_stats.get("success_rate", "0%"),
            avg_response_time_ms=qa_stats.get("avg_response_time_ms", 0),
            question_types=qa_stats.get("question_types", {})
        )
        
    except Exception as e:
        logger.error(f"获取统计错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# 问答统计端点
@router.get("/stats/qa")
async def get_qa_stats():
    """获取问答统计详细信息"""
    try:
        stats_instance = get_stats_instance()
        summary = stats_instance.get_summary()
        type_dist = stats_instance.get_question_type_distribution()
        recent = stats_instance.get_recent_questions(20)
        unanswered = stats_instance.get_unanswered_questions()
        
        return {
            "summary": summary,
            "question_type_distribution": type_dist,
            "recent_questions": recent,
            "unanswered_questions": unanswered
        }
    except Exception as e:
        logger.error(f"获取问答统计错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stats/clear")
async def clear_stats():
    """清空问答统计数据"""
    try:
        stats_instance = get_stats_instance()
        stats_instance.clear_stats()
        return {"status": "success", "message": "统计数据已清空"}
    except Exception as e:
        logger.error(f"清空统计错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
