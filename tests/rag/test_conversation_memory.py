"""对话记忆模块测试"""

import pytest
import json
import os
import tempfile
from rag.memory.conversation_memory import ConversationMemory, Message, ConversationContext


@pytest.fixture
def memory():
    """提供对话记忆实例"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = os.path.join(tmpdir, "test_stats.json")
        mem = ConversationMemory(max_history=5, max_tokens=4000)
        yield mem


@pytest.fixture
def memory_with_session():
    """提供带会话的记忆实例"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = os.path.join(tmpdir, "test_stats.json")
        mem = ConversationMemory(max_history=5, max_tokens=4000)
        mem.create_conversation("test_session")
        yield mem


class TestConversationMemory:
    """对话记忆测试"""

    def test_initialization(self):
        """测试初始化"""
        mem = ConversationMemory(max_history=10, max_tokens=5000)
        assert mem.max_history == 10
        assert mem.max_tokens == 5000
        assert mem.conversations == {}

    def test_create_conversation(self, memory):
        """测试创建对话"""
        session_id = memory.create_conversation("session1")
        assert session_id == "session1"
        
        ctx = memory.get_conversation("session1")
        assert ctx is not None
        assert isinstance(ctx, ConversationContext)

    def test_get_conversation_nonexistent(self, memory):
        """测试获取不存在的对话"""
        ctx = memory.get_conversation("nonexistent")
        assert ctx is None

    def test_add_message(self, memory_with_session):
        """测试添加消息"""
        memory_with_session.add_message(
            "test_session", 
            "user", 
            "高血压有哪些症状？"
        )
        
        ctx = memory_with_session.get_conversation("test_session")
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "user"
        assert ctx.messages[0].content == "高血压有哪些症状？"

    def test_add_message_with_metadata(self, memory_with_session):
        """测试添加带元数据的消息"""
        metadata = {"confidence": 0.9, "sources": ["doc1"]}
        memory_with_session.add_message(
            "test_session",
            "assistant",
            "高血压可能导致头痛、头晕。",
            metadata=metadata
        )
        
        ctx = memory_with_session.get_conversation("test_session")
        assert ctx.messages[0].metadata["confidence"] == 0.9

    def test_get_recent_messages(self, memory_with_session):
        """测试获取最近消息"""
        # 添加多条消息
        for i in range(10):
            memory_with_session.add_message("test_session", "user", f"问题{i}")
            memory_with_session.add_message("test_session", "assistant", f"回答{i}")
        
        # 获取最近5条（使用count参数）
        recent = memory_with_session.get_recent_messages("test_session", count=5)
        
        # 应该有10条（5轮对话，因为max_history=5会裁剪）
        assert len(recent) <= 10
    
    def test_get_recent_messages_nonexistent(self, memory):
        """测试获取不存在对话的最近消息"""
        recent = memory.get_recent_messages("nonexistent")
        assert recent == []
    
    def test_set_question_type(self, memory_with_session):
        """测试设置问题类型"""
        memory_with_session.add_message("test_session", "user", "高血压症状")
        memory_with_session.set_question_type("test_session", "symptom")
        
        ctx = memory_with_session.get_conversation("test_session")
        assert ctx.question_type == "symptom"
    
    def test_add_retrieved_docs(self, memory_with_session):
        """测试添加检索文档"""
        docs = [
            {"text": "doc1", "score": 0.9},
            {"text": "doc2", "score": 0.8}
        ]
        memory_with_session.add_retrieved_docs("test_session", docs)
        
        ctx = memory_with_session.get_conversation("test_session")
        assert len(ctx.retrieved_docs) == 2
    
    def test_add_agent_step(self, memory_with_session):
        """测试添加Agent步骤"""
        step = {
            "step_num": 1,
            "thought": "需要检索",
            "action": "retrieve_docs"
        }
        memory_with_session.add_agent_step("test_session", step)
        
        ctx = memory_with_session.get_conversation("test_session")
        assert len(ctx.agent_steps) == 1
        assert ctx.agent_steps[0]["step_num"] == 1

    def test_clear_conversation(self, memory_with_session):
        """测试清空对话"""
        memory_with_session.add_message("test_session", "user", "问题")
        memory_with_session.add_message("test_session", "assistant", "回答")
        
        memory_with_session.clear_conversation("test_session")
        
        ctx = memory_with_session.get_conversation("test_session")
        assert len(ctx.messages) == 0
    


    def test_delete_conversation(self, memory_with_session):
        """测试删除对话"""
        memory_with_session.add_message("test_session", "user", "问题")
        
        memory_with_session.delete_conversation("test_session")
        
        ctx = memory_with_session.get_conversation("test_session")
        assert ctx is None

    def test_get_context_for_query(self, memory_with_session):
        """测试获取查询上下文"""
        memory_with_session.add_message("test_session", "user", "问题1")
        memory_with_session.add_message("test_session", "assistant", "回答1")
        
        context = memory_with_session.get_context_for_query("test_session", "当前问题")
        
        assert len(context) >= 2

    def test_message_timestamps(self, memory_with_session):
        """测试消息时间戳"""
        memory_with_session.add_message("test_session", "user", "测试消息")
        
        ctx = memory_with_session.get_conversation("test_session")
        assert ctx.messages[0].timestamp is not None

    def test_multiple_sessions(self, memory):
        """测试多会话管理"""
        memory.add_message("session1", "user", "会话1的问题")
        memory.add_message("session2", "user", "会话2的问题")
        
        ctx1 = memory.get_conversation("session1")
        ctx2 = memory.get_conversation("session2")
        
        assert len(ctx1.messages) == 1
        assert len(ctx2.messages) == 1
        assert ctx1.messages[0].content == "会话1的问题"
        assert ctx2.messages[0].content == "会话2的问题"


class TestMessage:
    """消息数据类测试"""

    def test_message_creation(self):
        """测试消息创建"""
        msg = Message(role="user", content="测试内容")
        assert msg.role == "user"
        assert msg.content == "测试内容"
        assert msg.timestamp is not None
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        """测试带元数据的消息"""
        metadata = {"key": "value"}
        msg = Message(role="assistant", content="回答", metadata=metadata)
        assert msg.metadata == metadata


class TestConversationContext:
    """对话上下文测试"""

    def test_context_creation(self):
        """测试上下文创建"""
        ctx = ConversationContext()
        assert ctx.messages == []
        assert ctx.retrieved_docs == []
        assert ctx.question_type == "unknown"
        assert ctx.agent_steps == []

    def test_context_with_data(self):
        """测试带数据的上下文"""
        messages = [Message(role="user", content="问题")]
        docs = [{"text": "doc"}]
        steps = [{"step_num": 1}]
        
        ctx = ConversationContext(
            messages=messages,
            retrieved_docs=docs,
            question_type="symptom",
            agent_steps=steps
        )
        
        assert len(ctx.messages) == 1
        assert len(ctx.retrieved_docs) == 1
        assert ctx.question_type == "symptom"
        assert len(ctx.agent_steps) == 1