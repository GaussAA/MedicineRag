"""API客户端模块 - 前后端分离

通过HTTP请求与后端API通信，实现真正的分离架构
"""

import json
import streamlit as st
import requests
from typing import Optional, List, Dict, Any, Generator
from urllib.parse import urljoin

from backend.config import config
from app.constants import DEFAULT_API_BASE_URL, DEFAULT_TIMEOUT, STREAM_TIMEOUT, UPLOAD_TIMEOUT


class APIClient:
    """API客户端（优化版）"""
    
    # 默认超时配置（保留类常量以保持向后兼容）
    DEFAULT_TIMEOUT = DEFAULT_TIMEOUT
    STREAM_TIMEOUT = STREAM_TIMEOUT
    UPLOAD_TIMEOUT = UPLOAD_TIMEOUT
    
    def __init__(self, base_url: str = None):
        """初始化API客户端
        
        Args:
            base_url: API基础地址，默认 http://localhost:8000
        """
        self.base_url = base_url or DEFAULT_API_BASE_URL
        self.session = requests.Session()
        
        # 添加连接池和重试策略
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置默认请求头
        self.session.headers.update({
            "Accept": "application/json"
        })
    
    def _get_url(self, path: str) -> str:
        """获取完整URL"""
        return urljoin(self.base_url, path)
    
    # ==================== SSE 解析辅助方法 ====================

    def _parse_sse_stream(
        self,
        response: requests.Response,
        yield_content: bool = True,
        yield_sources: bool = True,
        yield_followup: bool = True,
        yield_knowledge_gaps: bool = True,
        yield_confidence: bool = True,
        yield_steps: bool = False
    ) -> Generator[str, None, None]:
        """解析SSE流式响应

        Args:
            response: requests.Response对象
            yield_content: 是否yield内容
            yield_sources: 是否yield来源
            yield_followup: 是否yield追问
            yield_knowledge_gaps: 是否yield知识缺口
            yield_confidence: 是否yield置信度
            yield_steps: 是否yield推理步骤

        Yields:
            流式返回的数据片段
        """
        buffer = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                buffer += chunk

                while '\n\n' in buffer:
                    line, buffer = buffer.split('\n\n', 1)
                    if line.startswith('data: '):
                        data = line[6:]

                        # 处理JSON格式的SSE数据
                        if data.startswith('{') and data.endswith('}'):
                            try:
                                json_data = json.loads(data)
                                msg_type = json_data.get('type')

                                if msg_type == 'error':
                                    yield f"\n\n错误: {json_data.get('message', '未知错误')}"
                                    return
                                elif msg_type == 'done':
                                    return
                                elif msg_type == 'source' and yield_sources:
                                    yield f"__SOURCE__: {json.dumps(json_data.get('data', {}))}"
                                elif msg_type == 'content' and yield_content and json_data.get('content'):
                                    yield json_data['content']
                                elif msg_type == 'followup' and yield_followup:
                                    yield f"__FOLLOWUP__: {json.dumps(json_data.get('data', []))}"
                                elif msg_type == 'knowledge_gap' and yield_knowledge_gaps:
                                    yield f"__KNOWLEDGE_GAPS__: {json.dumps(json_data.get('data', []))}"
                                elif msg_type == 'confidence' and yield_confidence:
                                    yield f"__CONFIDENCE__: {json.dumps(json_data.get('data', 0.0))}"
                                elif msg_type == 'step' and yield_steps:
                                    yield f"__STEPS__: {json.dumps(json_data.get('data', {}))}"
                                elif msg_type == 'steps' and yield_steps:
                                    yield f"__STEPS__: {json.dumps(json_data.get('data', []))}"
                                elif msg_type == 'disclaimer':
                                    yield f"__DISCLAIMER__: {json.dumps(json_data.get('data', ''))}"
                            except json.JSONDecodeError:
                                pass
                        # 处理纯文本
                        elif data and data != '[DONE]':
                            yield data

    def _handle_request_error(self, error: Exception) -> str:
        """处理请求错误，返回错误消息

        Args:
            error: 异常对象

        Returns:
            错误消息字符串
        """
        if isinstance(error, requests.exceptions.ConnectionError):
            return "无法连接到后端服务，请确保API服务正在运行。"
        elif isinstance(error, requests.exceptions.Timeout):
            return "请求超时，请稍后重试。"
        elif isinstance(error, requests.exceptions.HTTPError):
            try:
                error_detail = error.response.json().get("detail", str(error))
            except Exception:
                error_detail = str(error)
            return f"服务器错误: {error_detail}"
        else:
            return f"发生错误: {str(error)}"

    # ==================== 问答相关 ====================

    def ask_stream(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        """流式问答

        Args:
            question: 用户问题
            history: 对话历史

        Yields:
            流式返回的回答片段
        """
        url = self._get_url("/api/qa/stream")
        payload = {
            "question": question,
            "history": history or []
        }

        try:
            response = self.session.request(
                "POST", url, json=payload,
                timeout=self.STREAM_TIMEOUT,
                stream=True
            )
            response.raise_for_status()

            # 使用统一的SSE解析方法
            yield from self._parse_sse_stream(
                response,
                yield_sources=True,
                yield_followup=False,
                yield_knowledge_gaps=False,
                yield_confidence=False,
                yield_steps=False
            )
        except requests.exceptions.ConnectionError:
            yield "无法连接到后端服务，请确保API服务正在运行。"
        except requests.exceptions.Timeout:
            yield "请求超时，请稍后重试。"
        except Exception as e:
            yield f"发生错误: {str(e)}"
    
    def ask(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """非流式问答
        
        Args:
            question: 用户问题
            history: 对话历史
            
        Returns:
            问答响应字典
        """
        url = self._get_url("/api/qa/ask")
        payload = {
            "question": question,
            "history": history or []
        }
        
        response = self.session.post(url, json=payload, timeout=self.STREAM_TIMEOUT)
        response.raise_for_status()
        return response.json()
    
    # ==================== Agent 问答 ====================
    
    def ask_agent_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        enable_followup: bool = True,
        enable_knowledge_gap: bool = True
    ) -> Generator[str, None, None]:
        """流式 Agent 问答

        Args:
            question: 用户问题
            history: 对话历史
            session_id: 会话ID
            enable_followup: 是否启用追问
            enable_knowledge_gap: 是否启用知识缺口识别

        Yields:
            流式返回的回答片段
        """
        url = self._get_url("/api/qa/agent/stream")
        payload = {
            "question": question,
            "history": history or [],
            "session_id": session_id,
            "enable_followup": enable_followup,
            "enable_knowledge_gap": enable_knowledge_gap
        }

        try:
            response = self.session.request(
                "POST", url, json=payload,
                timeout=self.STREAM_TIMEOUT,
                stream=True
            )
            response.raise_for_status()

            # 使用统一的SSE解析方法（Agent模式支持所有消息类型）
            yield from self._parse_sse_stream(
                response,
                yield_sources=True,
                yield_followup=True,
                yield_knowledge_gaps=True,
                yield_confidence=True,
                yield_steps=True
            )
        except requests.exceptions.ConnectionError:
            yield "无法连接到后端服务，请确保API服务正在运行。"
        except requests.exceptions.Timeout:
            yield "请求超时，请稍后重试。"
        except Exception as e:
            yield f"发生错误: {str(e)}"
    
    def ask_agent(
        self, 
        question: str, 
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        enable_followup: bool = True,
        enable_knowledge_gap: bool = True
    ) -> Dict[str, Any]:
        """非流式 Agent 问答
        
        Args:
            question: 用户问题
            history: 对话历史
            session_id: 会话ID
            enable_followup: 是否启用追问
            enable_knowledge_gap: 是否启用知识缺口识别
            
        Returns:
            Agent 问答响应字典
        """
        url = self._get_url("/api/qa/agent")
        payload = {
            "question": question,
            "history": history or [],
            "session_id": session_id,
            "enable_followup": enable_followup,
            "enable_knowledge_gap": enable_knowledge_gap
        }
        
        response = self.session.post(url, json=payload, timeout=self.STREAM_TIMEOUT)
        response.raise_for_status()
        return response.json()
    
    # ==================== 文档相关 ====================
    
    def upload_document(self, file_path: str, file_name: str = None) -> Dict[str, Any]:
        """上传文档
        
        Args:
            file_path: 文件路径
            file_name: 文件名（可选）
            
        Returns:
            上传结果
        """
        url = self._get_url("/api/docs/upload")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_name or 'document', f)}
            response = self.session.post(url, files=files, timeout=self.UPLOAD_TIMEOUT)
        
        response.raise_for_status()
        return response.json()
    
    def upload_document_from_uploaded(self, uploaded_file, file_name: str) -> Dict[str, Any]:
        """从Streamlit上传文件对象创建临时文件并上传
        
        Args:
            uploaded_file: Streamlit的上传文件对象
            file_name: 文件名
            
        Returns:
            上传结果
        """
        import tempfile
        import os
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        try:
            return self.upload_document(tmp_path, file_name)
        finally:
            # 清理临时文件
            os.unlink(tmp_path)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """获取文档列表
        
        Returns:
            文档列表
        """
        url = self._get_url("/api/docs/list")
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("documents", [])
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            删除结果
        """
        url = self._get_url(f"/api/docs/{doc_id}")
        response = self.session.delete(url)
        response.raise_for_status()
        return response.json()
    
    def rebuild_index(self) -> Dict[str, Any]:
        """重建索引（异步后台处理）
        
        Returns:
            重建结果（立即返回，后台处理）
        """
        url = self._get_url("/api/docs/rebuild")
        response = self.session.post(url, timeout=self.DEFAULT_TIMEOUT)  # 短超时，只等待启动
        response.raise_for_status()
        return response.json()
    
    def get_rebuild_status(self) -> Dict[str, Any]:
        """获取重建索引状态
        
        Returns:
            重建任务状态
        """
        url = self._get_url("/api/docs/rebuild/status")
        response = self.session.get(url, timeout=self.DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """清空知识库
        
        Returns:
            清空结果
        """
        url = self._get_url("/api/docs/clear")
        response = self.session.post(url)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计
        
        Returns:
            统计信息
        """
        url = self._get_url("/api/docs/stats")
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_qa_stats(self) -> Dict[str, Any]:
        """获取问答统计详细信息
        
        Returns:
            问答统计详情
        """
        url = self._get_url("/api/docs/stats/qa")
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def clear_stats(self) -> Dict[str, Any]:
        """清空统计数据
        
        Returns:
            操作结果
        """
        url = self._get_url("/api/docs/stats/clear")
        response = self.session.post(url)
        response.raise_for_status()
        return response.json()
    
    # ==================== 健康检查 ====================
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            健康状态
        """
        url = self._get_url("/health")
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def detailed_health_check(self) -> Dict[str, Any]:
        """详细健康检查
        
        Returns:
            详细健康状态
        """
        url = self._get_url("/health/detailed")
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


import streamlit as st

from backend.config import config

# 默认API地址
DEFAULT_API_BASE_URL = "http://localhost:8000"


@st.cache_resource
def get_api_client() -> APIClient:
    """获取API客户端单例（使用Streamlit缓存）"""
    return APIClient()


def reset_api_client(base_url: str = None):
    """重置API客户端
    
    Args:
        base_url: 新的API地址
    """
    # 清除缓存后重新获取
    st.cache_resource.clear()
    return get_api_client()
