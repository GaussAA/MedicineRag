"""Agent 抽象基类

定义 Agent 的通用接口和行为模式。
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from backend.logging_config import get_logger

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent 状态"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class AgentStep:
    """Agent 推理步骤"""
    step_num: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    reflection: Optional[str] = None
    state: AgentState = AgentState.THINKING


@dataclass
class AgentResult:
    """Agent 执行结果"""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    warning: Optional[str] = None
    disclaimer: str = "本回答仅供参考，不能替代专业医疗建议。如有严重症状，请及时就医。"
    steps: List[AgentStep] = field(default_factory=list)
    followup_questions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    requires_followup: bool = False


class BaseAgent(ABC):
    """Agent 抽象基类
    
    提供 ReAct (Reasoning + Acting) 推理模式的基础框架。
    """

    def __init__(
        self,
        max_steps: int = 10,
        temperature: float = 0.2,
        timeout: int = 300,
    ):
        """初始化 Agent
        
        Args:
            max_steps: 最大推理步骤数
            temperature: LLM 温度参数
            timeout: 超时时间（秒）
        """
        self.max_steps = max_steps
        self.temperature = temperature
        self.timeout = timeout
        self.state = AgentState.IDLE
        self.steps: List[AgentStep] = []
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: List[Dict[str, Any]] = []

    @property
    def tools(self) -> Dict[str, Callable]:
        """获取已注册的工具"""
        return self._tools

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        """获取工具 schema 定义"""
        return self._tool_schemas

    def register_tool(self, name: str, func: Callable, description: str, parameters: Dict[str, Any]) -> None:
        """注册工具
        
        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
            parameters: 参数 schema
        """
        self._tools[name] = func
        self._tool_schemas.append({
            "name": name,
            "description": description,
            "parameters": parameters
        })
        logger.info(f"注册工具: {name}")

    def unregister_tool(self, name: str) -> None:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            self._tool_schemas = [s for s in self._tool_schemas if s["name"] != name]
            logger.info(f"注销工具: {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        return self._tools.get(name)

    def reset(self) -> None:
        """重置 Agent 状态"""
        self.state = AgentState.IDLE
        self.steps = []

    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """执行 Agent 推理
        
        Args:
            query: 用户查询
            context: 额外上下文信息
            
        Returns:
            AgentResult: 执行结果
        """
        self.reset()
        self.state = AgentState.THINKING
        
        try:
            result = await self._run_react_loop(query, context or {})
            self.state = AgentState.FINISHED
            return result
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}", exc_info=True)
            self.state = AgentState.ERROR
            return AgentResult(
                answer=f"处理您的问题时出现错误: {str(e)}",
                confidence=0.0
            )

    @abstractmethod
    async def _run_react_loop(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """ReAct 推理循环 - 子类实现
        
        Args:
            query: 用户查询
            context: 上下文
            
        Returns:
            AgentResult: 执行结果
        """
        pass

    @abstractmethod
    async def _think(self, query: str, context: Dict[str, Any]) -> str:
        """思考步骤 - 子类实现
        
        分析问题，决定下一步行动
        
        Args:
            query: 用户查询
            context: 当前上下文
            
        Returns:
            str: 思考结果
        """
        pass

    @abstractmethod
    async def _act(self, thought: str, context: Dict[str, Any]) -> str:
        """行动步骤 - 子类实现
        
        执行工具调用
        
        Args:
            thought: 思考结果
            context: 当前上下文
            
        Returns:
            str: 行动观察结果
        """
        pass

    @abstractmethod
    async def _reflect(self, query: str, context: Dict[str, Any]) -> str:
        """反思步骤 - 子类实现
        
        检查当前结果是否足够回答问题
        
        Args:
            query: 用户查询
            context: 当前上下文
            
        Returns:
            str: 反思结果
        """
        pass

    def _add_step(
        self,
        thought: str,
        action: Optional[str] = None,
        action_input: Optional[Dict[str, Any]] = None,
        observation: Optional[str] = None,
        reflection: Optional[str] = None,
        state: AgentState = AgentState.THINKING
    ) -> AgentStep:
        """添加推理步骤"""
        step = AgentStep(
            step_num=len(self.steps) + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            reflection=reflection,
            state=state
        )
        self.steps.append(step)
        return step

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tools_desc = "\n".join([
            f"- {s['name']}: {s['description']}"
            for s in self._tool_schemas
        ])
        
        return f"""你是一个专业的医疗问答助手。你可以使用以下工具来帮助回答用户的问题：

{tools_desc}

请按照以下步骤思考和回答：
1. 分析用户问题的意图和类型
2. 如需检索知识库，使用 retrieve_docs 工具
3. 如需检查安全性，使用 check_security 工具
4. 检查检索结果是否足够回答问题
5. 如不充分，考虑重新检索或追问

注意：
- 必须给出专业、准确的回答
- 如不确定，请明确告知用户
- 提供免责声明
- 如发现问题不完整，可主动追问"""
