"""Agent工具测试"""

import pytest
import json
from unittest.mock import Mock, patch

from rag.agents.tools.retriever_tool import RetrieverTool, RetrievalResult, create_retriever_tool
from rag.agents.tools.security_tool import SecurityTool, create_security_tool
from rag.agents.tools.knowledge_gap_tool import KnowledgeGapTool, create_knowledge_gap_tool
from rag.agents.tools.followup_tool import FollowUpTool, create_followup_tool


class MockRAGEngine:
    """Mock RAG引擎"""
    
    def retrieve(self, query: str, top_k: int = 5, use_hybrid: bool = True):
        return [
            {
                "text": f"Document about {query} - content {i}",
                "score": 0.95 - i * 0.1,
                "metadata": {"chunk_id": i, "title": f"Test Doc {i}"}
            }
            for i in range(min(3, top_k))
        ]


class MockSecurityService:
    """Mock安全服务"""
    
    def check_content(self, text: str):
        from backend.services.security_service import CheckResult
        
        if any(word in text for word in ["自杀", "死亡", "杀人"]):
            return CheckResult(
                is_safe=False,
                category="violence",
                warning_message="包含敏感内容"
            )
        return CheckResult(
            is_safe=True,
            category=None,
            warning_message=None
        )
    
    def is_emergency_symptom(self, text: str) -> bool:
        """检查是否是紧急症状"""
        emergency_keywords = ["胸痛", "呼吸困难", "大出血", "休克", "中风", "心脏病"]
        return any(word in text for word in emergency_keywords)
    
    def get_emergency_message(self) -> str:
        """获取紧急警告消息"""
        return "⚠️ 警告：您描述的症状可能涉及紧急情况，请立即拨打120急救电话或前往最近医院就诊！"


# ==================== RetrieverTool 测试 ====================

class TestRetrieverTool:
    """检索工具测试"""
    
    @pytest.fixture
    def mock_engine(self):
        return MockRAGEngine()
    
    @pytest.fixture
    def retriever(self, mock_engine):
        return RetrieverTool(mock_engine)
    
    def test_retriever_initialization(self, retriever):
        """测试检索工具初始化"""
        assert retriever is not None
        assert retriever.rag_engine is not None
    
    def test_retrieve_basic(self, retriever):
        """测试基本检索功能"""
        result = retriever.retrieve("高血压")
        
        assert result is not None
        # 解析JSON结果
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert result_dict["status"] == "success"
        assert result_dict["total"] > 0
    
    def test_retrieve_with_top_k(self, retriever):
        """测试指定top_k检索"""
        result = retriever.retrieve("糖尿病", top_k=2)
        
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert len(result_dict["documents"]) <= 2
    
    def test_retrieve_with_search_type(self, retriever):
        """测试指定搜索类型"""
        result = retriever.retrieve("头痛", search_type="symptom")
        
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert result_dict["status"] == "success"
    
    def test_get_schema(self, retriever):
        """测试获取工具schema"""
        schema = retriever.get_schema()
        
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
    
    def test_create_retriever_tool(self):
        """测试创建检索工具工厂函数"""
        engine = MockRAGEngine()
        tool = create_retriever_tool(engine)
        
        assert isinstance(tool, RetrieverTool)


# ==================== SecurityTool 测试 ====================

class TestSecurityTool:
    """安全检查工具测试"""
    
    @pytest.fixture
    def mock_service(self):
        return MockSecurityService()
    
    @pytest.fixture
    def security_tool(self, mock_service):
        return SecurityTool(mock_service)
    
    def test_security_tool_initialization(self, security_tool):
        """测试安全工具初始化"""
        assert security_tool is not None
    
    def test_check_safe_content(self, security_tool):
        """测试安全内容检查"""
        result = security_tool.check("高血压有哪些症状？")
        
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert result_dict["status"] == "success"
        assert result_dict["is_safe"] is True
    
    def test_check_unsafe_content(self, security_tool):
        """测试不安全内容检查"""
        result = security_tool.check("我想自杀怎么办")
        
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert result_dict["status"] == "success"
        assert result_dict["is_safe"] is False
    
    def test_get_schema(self, security_tool):
        """测试获取工具schema"""
        schema = security_tool.get_schema()
        
        assert "name" in schema
        assert "description" in schema
    
    def test_create_security_tool(self):
        """测试创建安全工具工厂函数"""
        service = MockSecurityService()
        tool = create_security_tool(service)
        
        assert isinstance(tool, SecurityTool)


# ==================== KnowledgeGapTool 测试 ====================

class TestKnowledgeGapTool:
    """知识缺口工具测试"""
    
    @pytest.fixture
    def gap_tool(self):
        return KnowledgeGapTool()
    
    def test_gap_tool_initialization(self, gap_tool):
        """测试知识缺口工具初始化"""
        assert gap_tool is not None
    
    def test_identify_gap_with_docs(self, gap_tool):
        """测试有文档时的知识缺口识别"""
        result = gap_tool.identify_gaps(
            query="高血压的治疗",
            retrieved_docs=[
                {"text": "高血压需要服用降压药", "score": 0.9},
                {"text": "定期测量血压很重要", "score": 0.8}
            ],
            confidence=0.85
        )
        
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert "gaps" in result_dict
    
    def test_identify_gap_no_docs(self, gap_tool):
        """测试无文档时的知识缺口识别"""
        result = gap_tool.identify_gaps(
            query="某种罕见病",
            retrieved_docs=[],
            confidence=0.0
        )
        
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert "gaps" in result_dict
        # 无文档时应该识别出知识缺口
        assert len(result_dict["gaps"]) > 0
    
    def test_identify_gap_low_relevance(self, gap_tool):
        """测试低相关度时的知识缺口识别"""
        result = gap_tool.identify_gaps(
            query="心脏病",
            retrieved_docs=[
                {"text": "不相关内容", "score": 0.2},
                {"text": "另一不相关内容", "score": 0.15}
            ],
            confidence=0.3
        )
        
        result_dict = json.loads(result) if isinstance(result, str) else result
        # 低相关度应该识别出知识缺口
        assert "gaps" in result_dict
    
    def test_get_schema(self, gap_tool):
        """测试获取工具schema"""
        schema = gap_tool.get_schema()
        
        assert "name" in schema
        assert "description" in schema
    
    def test_create_knowledge_gap_tool(self):
        """测试创建知识缺口工具工厂函数"""
        tool = create_knowledge_gap_tool()
        
        assert isinstance(tool, KnowledgeGapTool)


# ==================== FollowUpTool 测试 ====================

class TestFollowUpTool:
    """追问工具测试"""
    
    @pytest.fixture
    def followup_tool(self):
        return FollowUpTool()
    
    def test_followup_tool_initialization(self, followup_tool):
        """测试追问工具初始化"""
        assert followup_tool is not None
    
    def test_generate_followup_basic(self, followup_tool):
        """测试基本追问生成"""
        result = followup_tool.generate_questions(
            query="高血压有哪些症状？"
        )
        
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert "followup_questions" in result_dict
    
    def test_generate_followup_with_context(self, followup_tool):
        """测试带上下文的追问生成"""
        result = followup_tool.generate_questions(
            query="糖尿病的诊断标准",
            context="diagnosis"
        )
        
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert "followup_questions" in result_dict
    
    def test_generate_followup_various_types(self, followup_tool):
        """测试不同类型问题的追问"""
        test_queries = [
            "头痛怎么办",
            "吃什么药",
            "怎么诊断"
        ]
        
        for query in test_queries:
            result = followup_tool.generate_questions(query)
            result_dict = json.loads(result) if isinstance(result, str) else result
            assert "followup_questions" in result_dict
    
    def test_get_schema(self, followup_tool):
        """测试获取工具schema"""
        schema = followup_tool.get_schema()
        
        assert "name" in schema
        assert "description" in schema
    
    def test_create_followup_tool(self):
        """测试创建追问工具工厂函数"""
        tool = create_followup_tool()
        
        assert isinstance(tool, FollowUpTool)


# ==================== 工具集成测试 ====================

class TestToolsIntegration:
    """工具集成测试"""
    
    def test_all_tools_creation(self):
        """测试所有工具创建"""
        # 创建检索工具
        retriever = create_retriever_tool(MockRAGEngine())
        assert retriever is not None
        
        # 创建安全工具
        security = create_security_tool(MockSecurityService())
        assert security is not None
        
        # 创建知识缺口工具
        gap = create_knowledge_gap_tool()
        assert gap is not None
        
        # 创建追问工具
        followup = create_followup_tool()
        assert followup is not None
    
    def test_tools_schema_consistency(self):
        """测试工具schema一致性"""
        tools = [
            create_retriever_tool(MockRAGEngine()),
            create_security_tool(MockSecurityService()),
            create_knowledge_gap_tool(),
            create_followup_tool()
        ]
        
        for tool in tools:
            schema = tool.get_schema()
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
