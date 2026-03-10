"""MedicalAgent 测试"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import asdict

from rag.agents.base import AgentState, AgentConfig
from rag.agents.medical_agent import MedicalAgent, AgentConfig as MedicalAgentConfig


class MockRAGEngine:
    """Mock RAG引擎"""
    
    def __init__(self):
        self.retrieve_count = 0
    
    def retrieve(self, query: str, top_k: int = 5):
        self.retrieve_count += 1
        return [
            {
                "text": f"Mock document {i} about {query}",
                "score": 0.9 - i * 0.1,
                "metadata": {"chunk_id": i, "title": "Test Document"}
            }
            for i in range(min(3, top_k))
        ]
    
    def generate(self, query: str, context: list, **kwargs):
        return f"Generated answer for: {query}"


class MockSecurityService:
    """Mock安全服务"""
    
    def check(self, text: str) -> dict:
        if "自杀" in text or "死亡" in text:
            return {
                "is_safe": False,
                "warning": "包含敏感内容",
                "category": "emergency"
            }
        return {"is_safe": True, "warning": None, "category": None}


class MockQuestionTypeDetector:
    """Mock问题类型检测器"""
    
    def detect(self, question: str) -> str:
        if "症状" in question:
            return "symptom"
        elif "治疗" in question or "药物" in question:
            return "treatment"
        elif "诊断" in question:
            return "diagnosis"
        return "unknown"


class MockConfidenceCalculator:
    """Mock置信度计算器"""
    
    def calculate(self, docs: list, question_type: str = "unknown") -> float:
        if not docs:
            return 0.0
        return min(0.95, sum(d.get("score", 0) for d in docs) / len(docs))


@pytest.fixture
def mock_rag_engine():
    """创建Mock RAG引擎"""
    return MockRAGEngine()


@pytest.fixture
def mock_security_service():
    """创建Mock安全服务"""
    return MockSecurityService()


@pytest.fixture
def mock_question_type_detector():
    """创建Mock问题类型检测器"""
    return MockQuestionTypeDetector()


@pytest.fixture
def mock_confidence_calculator():
    """创建Mock置信度计算器"""
    return MockConfidenceCalculator()


@pytest.fixture
def medical_agent(
    mock_rag_engine,
    mock_security_service,
    mock_question_type_detector,
    mock_confidence_calculator
):
    """创建MedicalAgent实例"""
    config = MedicalAgentConfig(
        max_steps=3,
        timeout=60,
        enable_reflection=True,
        enable_followup=True,
        enable_knowledge_gap=True
    )
    return MedicalAgent(
        rag_engine=mock_rag_engine,
        security_service=mock_security_service,
        question_type_detector=mock_question_type_detector,
        confidence_calculator=mock_confidence_calculator,
        config=config
    )


class TestMedicalAgentInitialization:
    """MedicalAgent初始化测试"""
    
    def test_agent_creation(self, medical_agent):
        """测试Agent创建"""
        assert medical_agent is not None
        assert medical_agent.config.max_steps == 3
        assert medical_agent.config.enable_reflection is True
    
    def test_tools_registered(self, medical_agent):
        """测试工具注册"""
        tools = medical_agent.get_available_tools()
        
        assert "retrieve_docs" in tools
        assert "check_security" in tools
        assert "generate_followup_questions" in tools
        assert "identify_knowledge_gap" in tools
    
    def test_retriever_tool_exists(self, medical_agent):
        """测试检索工具存在"""
        assert hasattr(medical_agent, "retriever_tool")
        assert medical_agent.retriever_tool is not None
    
    def test_security_tool_exists(self, medical_agent):
        """测试安全工具存在"""
        assert hasattr(medical_agent, "security_tool")
        assert medical_agent.security_tool is not None


class TestMedicalAgentSecurity:
    """MedicalAgent安全检查测试"""
    
    @pytest.mark.asyncio
    async def test_safe_query(self, medical_agent):
        """测试安全查询"""
        result = await medical_agent.arun(
            "高血压有哪些症状？",
            session_id="test_session"
        )
        
        # 安全查询应该被处理
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_unsafe_query(self, medical_agent, mock_security_service):
        """测试不安全查询"""
        # 创建包含敏感词的查询
        result = await medical_agent.arun(
            "我想自杀怎么办",
            session_id="test_session_unsafe"
        )
        
        # 应该返回安全警告
        assert result is not None


class TestMedicalAgentRetrieval:
    """MedicalAgent检索测试"""
    
    @pytest.mark.asyncio
    async def test_retrieval(self, medical_agent, mock_rag_engine):
        """测试文档检索"""
        # 直接测试检索工具
        tool_func = medical_agent.get_tool("retrieve_docs")
        result = tool_func("高血压症状")
        
        assert result["status"] == "success"
        assert result["total"] > 0
        assert len(result["documents"]) > 0
    
    @pytest.mark.asyncio
    async def test_retrieval_with_context(self, medical_agent):
        """测试带上下文的检索"""
        result = await medical_agent.arun(
            "糖尿病的诊断标准",
            session_id="test_retrieval"
        )
        
        assert result is not None


class TestMedicalAgentFollowup:
    """MedicalAgent追问功能测试"""
    
    def test_followup_tool_exists(self, medical_agent):
        """测试追问工具存在"""
        assert hasattr(medical_agent, "followup_tool")
    
    def test_generate_followup_questions(self, medical_agent):
        """测试生成追问问题"""
        tool_func = medical_agent.get_tool("generate_followup_questions")
        
        result = tool_func(
            current_question="高血压有哪些症状？",
            answer="高血压可能导致头痛、头晕等症状。"
        )
        
        assert result is not None
        assert "followup_questions" in result


class TestMedicalAgentKnowledgeGap:
    """MedicalAgent知识缺口测试"""
    
    def test_knowledge_gap_tool_exists(self, medical_agent):
        """测试知识缺口工具存在"""
        assert hasattr(medical_agent, "knowledge_gap_tool")
    
    def test_identify_knowledge_gap(self, medical_agent):
        """测试识别知识缺口"""
        tool_func = medical_agent.get_tool("identify_knowledge_gap")
        
        result = tool_func(
            question="某种罕见病的治疗",
            retrieved_docs=[]
        )
        
        assert result is not None
        assert "knowledge_gaps" in result


class TestMedicalAgentIntegration:
    """MedicalAgent集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, medical_agent):
        """测试完整工作流"""
        result = await medical_agent.arun(
            "高血压的诊断标准是什么？",
            session_id="test_full"
        )
        
        assert result is not None
        # 检查返回结构
        result_dict = asdict(result) if hasattr(result, "__dict__") else result
        assert "answer" in result_dict or "steps" in result_dict
    
    @pytest.mark.asyncio
    async def test_symptom_question(self, medical_agent):
        """测试症状类问题"""
        result = await medical_agent.arun(
            "咳嗽有痰是什么原因？",
            session_id="test_symptom"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_treatment_question(self, medical_agent):
        """测试治疗类问题"""
        result = await medical_agent.arun(
            "糖尿病如何治疗？",
            session_id="test_treatment"
        )
        
        assert result is not None


class TestMedicalAgentConfig:
    """MedicalAgent配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MedicalAgentConfig()
        
        assert config.max_steps == 10
        assert config.temperature == 0.2
        assert config.timeout == 300
        assert config.enable_reflection is True
        assert config.enable_followup is True
        assert config.enable_knowledge_gap is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = MedicalAgentConfig(
            max_steps=5,
            temperature=0.5,
            timeout=120,
            enable_reflection=False
        )
        
        assert config.max_steps == 5
        assert config.temperature == 0.5
        assert config.timeout == 120
        assert config.enable_reflection is False
