"""Agent基础组件测试"""

import pytest
from dataclasses import asdict
from rag.agents.base import AgentState, AgentStep, AgentResult, BaseAgent
from rag.agents.medical_agent import AgentConfig


class MockAgent(BaseAgent):
    """用于测试的Mock Agent实现"""
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            max_steps=3,
            temperature=0.2,
            timeout=30
        )
        self._tools = {}
        self.call_count = 0
    
    def _run_react_loop(self, query: str, context: dict = None):
        """模拟ReAct循环"""
        self._add_step(
            thought="Mock thought",
            action="mock_action",
            observation="Mock observation"
        )
        return "Mock final answer"
    
    def _think(self, state: AgentState, context: dict) -> str:
        return "Thinking about the query"
    
    def _act(self, state: AgentState, action: str, context: dict) -> dict:
        self.call_count += 1
        return {"result": f"Action {action} executed", "call_num": self.call_count}
    
    def _reflect(self, state: AgentState, observation: str, context: dict) -> bool:
        return len(self.steps) >= self.max_steps


class TestAgentState:
    """Agent状态枚举测试"""
    
    def test_agent_state_values(self):
        """测试AgentState枚举值"""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.THINKING.value == "thinking"
        assert AgentState.ACTING.value == "acting"
        assert AgentState.REFLECTING.value == "reflecting"
        assert AgentState.FINISHED.value == "finished"
        assert AgentState.ERROR.value == "error"


class TestAgentStep:
    """Agent步骤测试"""
    
    def test_agent_step_creation(self):
        """测试AgentStep创建"""
        step = AgentStep(
            step_num=1,
            thought="思考过程",
            action="retrieve_docs",
            observation="找到3个文档",
            reflection="结果满意"
        )
        
        assert step.step_num == 1
        assert step.thought == "思考过程"
        assert step.action == "retrieve_docs"
        assert step.observation == "找到3个文档"
        assert step.reflection == "结果满意"
    
    def test_agent_step_to_dict(self):
        """测试AgentStep转字典"""
        step = AgentStep(
            step_num=1,
            thought="test thought",
            action="test_action",
            observation="test observation"
        )
        
        step_dict = asdict(step)
        assert step_dict["step_num"] == 1
        assert step_dict["thought"] == "test thought"


class TestAgentResult:
    """Agent结果测试"""
    
    def test_agent_result_creation(self):
        """测试AgentResult创建"""
        result = AgentResult(
            answer="测试回答",
            sources=[{"text": "source1", "score": 0.9}],
            confidence=0.85,
            steps=[]
        )
        
        assert result.answer == "测试回答"
        assert len(result.sources) == 1
        assert result.confidence == 0.85
    
    def test_agent_result_with_knowledge_gaps(self):
        """测试带知识缺口的AgentResult"""
        result = AgentResult(
            answer="部分回答",
            sources=[],
            confidence=0.5,
            steps=[],
            knowledge_gaps=["缺少心脏病相关知识"],
            requires_followup=True
        )
        
        assert result.knowledge_gaps == ["缺少心脏病相关知识"]
        assert result.requires_followup is True


class TestAgentConfig:
    """Agent配置测试（使用MedicalAgent的AgentConfig）"""
    
    def test_agent_config_from_medical_agent(self):
        """测试从MedicalAgent导入配置"""
        from rag.agents.medical_agent import AgentConfig as MedicalAgentConfig
        
        config = MedicalAgentConfig()
        
        # 使用配置默认值
        assert config.max_steps == 5  # 默认来自 config.AGENT_MAX_STEPS
        assert config.temperature == 0.1  # 默认来自 config.LLM_TEMPERATURE
        assert config.timeout == 300
        assert config.enable_reflection is True
        assert config.enable_followup is True
        assert config.enable_knowledge_gap is True
    
    def test_custom_agent_config(self):
        """测试自定义AgentConfig"""
        from rag.agents.medical_agent import AgentConfig as MedicalAgentConfig
        
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


class TestBaseAgent:
    """BaseAgent基类测试"""
    
    def test_agent_initialization(self):
        """测试Agent初始化"""
        agent = MockAgent()
        
        assert agent.max_steps == 3
        assert agent.temperature == 0.2
        assert agent.timeout == 30
        assert len(agent._tools) == 0
    
    def test_register_tool(self):
        """测试工具注册"""
        agent = MockAgent()
        
        def mock_tool(query: str):
            return {"result": "tool result"}
        
        agent.register_tool(
            "test_tool", 
            mock_tool,
            description="Test tool",
            parameters={"query": {"type": "string"}}
        )
        
        assert "test_tool" in agent._tools
        assert callable(agent._tools["test_tool"])
    
    def test_get_tool(self):
        """测试获取工具"""
        agent = MockAgent()
        
        def mock_tool(query: str):
            return {"result": "tool result"}
        
        agent.register_tool(
            "my_tool", 
            mock_tool,
            description="My tool",
            parameters={"query": {"type": "string"}}
        )
        retrieved_tool = agent.get_tool("my_tool")
        
        assert retrieved_tool is not None
        assert callable(retrieved_tool)
    
    def test_get_nonexistent_tool(self):
        """测试获取不存在的工具"""
        agent = MockAgent()
        
        result = agent.get_tool("nonexistent")
        
        assert result is None
    
    def test_tools_storage(self):
        """测试工具存储"""
        agent = MockAgent()
        
        def mock_tool(query: str):
            return {"result": "tool result"}
        
        agent.register_tool(
            "test_tool", 
            mock_tool,
            description="Test tool",
            parameters={"query": {"type": "string"}}
        )
        
        # 检查工具是否存储
        assert "test_tool" in agent._tools
        assert callable(agent._tools["test_tool"])
    
    def test_history_management(self):
        """测试历史记录管理"""
        agent = MockAgent()
        
        agent._add_step(
            thought="step 1",
            state=AgentState.THINKING
        )
        agent._add_step(
            thought="step 2", 
            action="test",
            observation="result",
            state=AgentState.ACTING
        )
        
        assert len(agent.steps) == 2
    
    def test_mock_react_loop(self):
        """测试MockAgent的ReAct循环"""
        agent = MockAgent()
        
        result = agent._run_react_loop("测试问题")
        
        assert result == "Mock final answer"
        assert len(agent.steps) == 1
