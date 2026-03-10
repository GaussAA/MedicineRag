"""安全检查工具

封装安全检查服务为 Agent 可调用的工具。
"""

import json
from typing import Any, Dict, Optional
from dataclasses import dataclass

from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityCheckResult:
    """安全检查结果"""
    is_safe: bool
    is_emergency: bool
    blocked_reason: Optional[str] = None
    emergency_warning: Optional[str] = None
    detected_categories: list = None

    def __post_init__(self):
        if self.detected_categories is None:
            self.detected_categories = []


class SecurityTool:
    """安全检查工具类"""

    def __init__(self, security_service):
        """初始化安全检查工具
        
        Args:
            security_service: 安全检查服务实例
        """
        self.security_service = security_service

    def check(self, query: str) -> str:
        """检查问题安全性
        
        Args:
            query: 用户问题
            
        Returns:
            str: 检查结果 JSON 字符串
        """
        try:
            # 使用 SecurityService 检查内容
            result = self.security_service.check_content(query)
            
            # 额外检查是否是紧急症状
            is_emergency = self.security_service.is_emergency_symptom(query)
            emergency_warning = self.security_service.get_emergency_message() if is_emergency else None
            
            response = {
                "status": "success",
                "is_safe": result.is_safe,
                "is_emergency": is_emergency,
                "detected_categories": [result.category] if result.category else []
            }
            
            if not result.is_safe:
                response["blocked_reason"] = result.warning_message
                
            if is_emergency:
                response["emergency_warning"] = emergency_warning
                
            logger.info(f"安全检查: query={query[:50]}..., safe={result.is_safe}, emergency={is_emergency}")
            return json.dumps(response, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"安全检查异常: {e}")
            return json.dumps({
                "status": "error",
                "message": f"安全检查失败: {str(e)}"
            }, ensure_ascii=False)

    def check_with_context(
        self,
        query: str,
        conversation_history: list
    ) -> str:
        """检查问题安全性（考虑对话上下文）
        
        Args:
            query: 当前问题
            conversation_history: 对话历史
            
        Returns:
            str: 检查结果
        """
        try:
            # 构建完整上下文
            full_context = query
            if conversation_history:
                history_text = " | ".join([
                    item.get("content", "") 
                    for item in conversation_history[-3:]
                ])
                full_context = f"{history_text} | {query}"
            
            return self.check(full_context)
            
        except Exception as e:
            logger.error(f"安全检查异常: {e}")
            return json.dumps({
                "status": "error",
                "message": f"安全检查失败: {str(e)}"
            }, ensure_ascii=False)

    def is_emergency(self, query: str) -> str:
        """快速检查是否为紧急情况
        
        Args:
            query: 用户问题
            
        Returns:
            str: 检查结果
        """
        try:
            is_emergency = self.security_service.is_emergency_symptom(query)
            emergency_warning = self.security_service.get_emergency_message() if is_emergency else None
            
            response = {
                "is_emergency": is_emergency,
                "warning": emergency_warning
            }
            
            return json.dumps(response, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"紧急检查异常: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            }, ensure_ascii=False)

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 schema"""
        return {
            "name": "check_security",
            "description": "检查用户问题是否涉及敏感内容或紧急医疗情况。如果用户问题涉及紧急症状（如胸痛、呼吸困难、严重出血等），必须使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要检查的用户问题"
                    }
                },
                "required": ["query"]
            }
        }


def create_security_tool(security_service) -> SecurityTool:
    """创建安全检查工具
    
    Args:
        security_service: 安全检查服务实例
        
    Returns:
        SecurityTool: 安全检查工具实例
    """
    return SecurityTool(security_service)
