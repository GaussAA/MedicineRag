"""对话记忆模块

管理 Agent 的对话历史和上下文。
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """对话消息"""
    role: str  # user/assistant/system
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """对话上下文"""
    messages: List[Message] = field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    question_type: str = "unknown"
    agent_steps: List[Dict[str, Any]] = field(default_factory=list)


class ConversationMemory:
    """对话记忆管理类
    
    管理多轮对话历史，支持上下文检索。
    """

    def __init__(
        self,
        max_history: int = 10,
        max_tokens: int = 4000
    ):
        """初始化对话记忆
        
        Args:
            max_history: 最大历史轮数
            max_tokens: 最大 token 数
        """
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.conversations: Dict[str, ConversationContext] = {}

    def create_conversation(self, session_id: str) -> str:
        """创建新对话
        
        Args:
            session_id: 会话 ID
            
        Returns:
            str: 会话 ID
        """
        self.conversations[session_id] = ConversationContext()
        logger.info(f"创建新对话: {session_id}")
        return session_id

    def get_conversation(self, session_id: str) -> Optional[ConversationContext]:
        """获取对话上下文
        
        Args:
            session_id: 会话 ID
            
        Returns:
            ConversationContext: 对话上下文，不存在返回 None
        """
        return self.conversations.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加消息
        
        Args:
            session_id: 会话 ID
            role: 角色 (user/assistant/system)
            content: 内容
            metadata: 额外元数据
        """
        if session_id not in self.conversations:
            self.create_conversation(session_id)
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.conversations[session_id].messages.append(message)
        
        # 裁剪历史
        self._trim_history(session_id)

    def add_retrieved_docs(
        self,
        session_id: str,
        docs: List[Dict[str, Any]]
    ) -> None:
        """添加检索文档记录
        
        Args:
            session_id: 会话 ID
            docs: 检索到的文档
        """
        if session_id in self.conversations:
            self.conversations[session_id].retrieved_docs = docs

    def set_question_type(
        self,
        session_id: str,
        question_type: str
    ) -> None:
        """设置问题类型
        
        Args:
            session_id: 会话 ID
            question_type: 问题类型
        """
        if session_id in self.conversations:
            self.conversations[session_id].question_type = question_type

    def add_agent_step(
        self,
        session_id: str,
        step: Dict[str, Any]
    ) -> None:
        """添加 Agent 推理步骤
        
        Args:
            session_id: 会话 ID
            step: 推理步骤
        """
        if session_id in self.conversations:
            self.conversations[session_id].agent_steps.append(step)

    def get_recent_messages(
        self,
        session_id: str,
        count: int = 5
    ) -> List[Dict[str, str]]:
        """获取最近的消息
        
        Args:
            session_id: 会话 ID
            count: 获取数量
            
        Returns:
            List[Dict[str, str]]: 消息列表
        """
        if session_id not in self.conversations:
            return []
        
        messages = self.conversations[session_id].messages
        return [
            {"role": m.role, "content": m.content}
            for m in messages[-count:]
        ]

    def get_context_for_query(
        self,
        session_id: str,
        current_query: str
    ) -> List[Dict[str, Any]]:
        """获取用于增强查询的上下文
        
        Args:
            session_id: 会话 ID
            current_query: 当前查询
            
        Returns:
            List[Dict[str, Any]]: 上下文消息
        """
        return self.get_recent_messages(session_id, self.max_history)

    def clear_conversation(self, session_id: str) -> None:
        """清空对话
        
        Args:
            session_id: 会话 ID
        """
        if session_id in self.conversations:
            self.conversations[session_id] = ConversationContext()
            logger.info(f"清空对话: {session_id}")

    def delete_conversation(self, session_id: str) -> None:
        """删除对话
        
        Args:
            session_id: 会话 ID
        """
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info(f"删除对话: {session_id}")

    def _trim_history(self, session_id: str) -> None:
        """裁剪历史记录
        
        Args:
            session_id: 会话 ID
        """
        if session_id not in self.conversations:
            return
        
        context = self.conversations[session_id]
        
        # 裁剪消息数量
        if len(context.messages) > self.max_history * 2:  # 包含 user 和 assistant
            context.messages = context.messages[-self.max_history * 2:]
        
        # 基于字符近似估算token并进行裁剪（约1.5字符/token）
        # 注意：这是近似估算，不使用tokenizer库以保持轻量
        self._trim_by_token_estimate(context)

    def _trim_by_token_estimate(self, context: "ConversationContext") -> None:
        """根据token估算值裁剪上下文
        
        使用字符数/1.5作为token数的近似估算，
        这是一个轻量级的实现，不需要额外的tokenizer库。
        
        Args:
            context: 对话上下文
        """
        if not context.messages:
            return
        
        # 计算当前token估算值
        total_chars = sum(len(m.content) for m in context.messages)
        estimated_tokens = int(total_chars / 1.5)
        
        # 如果超过限制，逐步裁剪
        max_tokens = self.max_history * 200  # 每轮对话约200 tokens
        if estimated_tokens <= max_tokens:
            return
        
        # 从最旧的消息开始裁剪
        while estimated_tokens > max_tokens and len(context.messages) > 2:
            # 移除最旧的消息（第一个）
            removed_content = context.messages[0].content
            context.messages = context.messages[1:]
            estimated_tokens -= int(len(removed_content) / 1.5)

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        total_conversations = len(self.conversations)
        total_messages = sum(
            len(c.messages) 
            for c in self.conversations.values()
        )
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "max_history": self.max_history,
            "sessions": list(self.conversations.keys())
        }

    def save_to_file(self, file_path: str) -> None:
        """保存对话历史到文件
        
        Args:
            file_path: 文件路径
        """
        data = {
            session_id: {
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "metadata": m.metadata
                    }
                    for m in ctx.messages
                ],
                "retrieved_docs_count": len(ctx.retrieved_docs),
                "question_type": ctx.question_type
            }
            for session_id, ctx in self.conversations.items()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"对话历史已保存: {file_path}")

    def load_from_file(self, file_path: str) -> None:
        """从文件加载对话历史
        
        Args:
            file_path: 文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for session_id, ctx_data in data.items():
                messages = [
                    Message(
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", datetime.now().isoformat()),
                        metadata=m.get("metadata", {})
                    )
                    for m in ctx_data.get("messages", [])
                ]
                self.conversations[session_id] = ConversationContext(
                    messages=messages,
                    question_type=ctx_data.get("question_type", "unknown")
                )
            
            logger.info(f"对话历史已加载: {file_path}")
        except FileNotFoundError:
            logger.warning(f"对话历史文件不存在: {file_path}")
        except Exception as e:
            logger.error(f"加载对话历史失败: {e}")


# 全局单例
_memory_instance: Optional[ConversationMemory] = None


def get_conversation_memory() -> ConversationMemory:
    """获取对话记忆单例"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance
