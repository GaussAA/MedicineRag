"""API数据模型定义"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ========== 问答相关模型 ==========

class QARequest(BaseModel):
    """问答请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="对话历史")


class QAResponse(BaseModel):
    """问答响应"""
    answer: str = Field(..., description="LLM生成的回答")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="参考来源")
    disclaimer: str = Field(..., description="免责声明")
    question_type: Optional[str] = Field(default=None, description="问题类型")
    confidence_level: str = Field(default="normal", description="置信度")


class StreamQARequest(BaseModel):
    """流式问答请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="对话历史")


# ========== 文档相关模型 ==========

class UploadResponse(BaseModel):
    """上传响应"""
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    file_name: Optional[str] = Field(default=None, description="文件名")
    doc_count: Optional[int] = Field(default=None, description="文档块数量")


class DocInfo(BaseModel):
    """文档信息"""
    doc_id: str = Field(..., description="文档ID")
    file_name: str = Field(..., description="文件名")
    chunk_count: int = Field(default=0, description="块数量")
    size: int = Field(default=0, description="文件大小")


class DocListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocInfo] = Field(default_factory=list, description="文档列表")


class DeleteResponse(BaseModel):
    """删除响应"""
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")


class RebuildResponse(BaseModel):
    """重建索引响应"""
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    doc_count: Optional[int] = Field(default=None, description="文档数量")


# ========== 统计相关模型 ==========

class StatsResponse(BaseModel):
    """统计响应"""
    total_questions: int = Field(default=0, description="总问题数")
    successful_answers: int = Field(default=0, description="成功回答数")
    success_rate: str = Field(default="0%", description="成功率")
    avg_response_time_ms: float = Field(default=0, description="平均响应时间")
    question_types: Dict[str, int] = Field(default_factory=dict, description="问题类型分布")


# ========== 错误响应 ==========

class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    detail: str = Field(..., description="错误详情")
    error_code: Optional[str] = Field(default=None, description="错误代码")
