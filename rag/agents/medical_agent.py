"""医疗问答 Agent

基于 ReAct 推理模式的医疗问答 Agent 实现。
"""

import json
import re
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from backend.logging_config import get_logger
from backend.config import config
from backend.services.question_type_detector import (
    get_question_type_detector,
    is_medical_related,
    detect_question_type
)

from rag.agents.base import BaseAgent, AgentResult, AgentState
from rag.agents.tools.retriever_tool import RetrieverTool, create_retriever_tool
from rag.agents.tools.security_tool import SecurityTool, create_security_tool
from rag.agents.tools.knowledge_gap_tool import KnowledgeGapTool, create_knowledge_gap_tool
from rag.agents.tools.followup_tool import FollowUpTool, create_followup_tool
from rag.memory import ConversationMemory, get_conversation_memory
from rag.llm_manager import LLMManager

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Agent 配置"""
    max_steps: int = field(default_factory=lambda: config.AGENT_MAX_STEPS)
    temperature: float = field(default_factory=lambda: config.LLM_TEMPERATURE)
    timeout: int = 300
    enable_reflection: bool = True
    enable_followup: bool = True
    enable_knowledge_gap: bool = True
    min_confidence_threshold: float = 0.5
    min_docs_for_answer: int = field(default_factory=lambda: config.MIN_DOCS_FOR_ANSWER)


class MedicalAgent(BaseAgent):
    """医疗问答 Agent
    
    使用 ReAct (Reasoning + Acting) 推理模式，
    支持工具调用、反思、主动追问和知识缺口识别。
    """

    def __init__(
        self,
        rag_engine,
        security_service,
        question_type_detector,
        confidence_calculator,
        config: Optional[AgentConfig] = None
    ):
        """初始化医疗问答 Agent
        
        Args:
            rag_engine: RAG 引擎
            security_service: 安全检查服务
            question_type_detector: 问题类型检测器
            confidence_calculator: 置信度计算器
            config: Agent 配置
        """
        super().__init__(
            max_steps=config.max_steps if config else 10,
            temperature=config.temperature if config else 0.2,
            timeout=config.timeout if config else 300
        )
        
        self.config = config or AgentConfig()
        self.rag_engine = rag_engine
        self.security_service = security_service
        self.question_type_detector = question_type_detector
        self.confidence_calculator = confidence_calculator
        self.memory: ConversationMemory = get_conversation_memory()
        
        # 初始化工具
        self._init_tools()

    def _init_tools(self) -> None:
        """初始化工具"""
        # 创建检索工具
        self.retriever_tool = create_retriever_tool(self.rag_engine)
        self.register_tool(
            "retrieve_docs",
            self._tool_retrieve,
            **{
                "description": self.retriever_tool.get_schema()["description"],
                "parameters": self.retriever_tool.get_schema()["parameters"]
            }
        )
        
        # 创建安全检查工具
        self.security_tool = create_security_tool(self.security_service)
        self.register_tool(
            "check_security",
            self._tool_security_check,
            **{
                "description": self.security_tool.get_schema()["description"],
                "parameters": self.security_tool.get_schema()["parameters"]
            }
        )
        
        # 创建追问工具
        self.followup_tool = create_followup_tool()
        self.register_tool(
            "generate_followup_questions",
            self._tool_generate_followup,
            **{
                "description": self.followup_tool.get_schema()["description"],
                "parameters": self.followup_tool.get_schema()["parameters"]
            }
        )
        
        # 创建知识缺口工具
        self.knowledge_gap_tool = create_knowledge_gap_tool()
        self.register_tool(
            "identify_knowledge_gap",
            self._tool_identify_gap,
            **{
                "description": self.knowledge_gap_tool.get_schema()["description"],
                "parameters": self.knowledge_gap_tool.get_schema()["parameters"]
            }
        )
        
        logger.info("MedicalAgent 工具初始化完成")

    async def _run_react_loop(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """ReAct 推理循环
        
        Args:
            query: 用户查询
            context: 上下文
            
        Returns:
            AgentResult: 执行结果
        """
        session_id = context.get("session_id", "default")
        
        # ===== 预判阶段：检查问题是否与医疗相关 =====
        question_type = detect_question_type(query)
        logger.info(f"问题类型检测: {question_type}")
        
        # 非医疗问题直接返回友好回答
        if not is_medical_related(query):
            logger.info(f"检测到非医疗问题类型: {question_type}，跳过检索直接回答")
            
            # 生成友好回答
            friendly_answer = self._generate_friendly_response(query, question_type)
            
            # 添加到记忆
            self.memory.add_message(session_id, "assistant", friendly_answer)
            
            return AgentResult(
                answer=friendly_answer,
                sources=[],
                steps=[],
                confidence=1.0
            )
        # ===== 预判阶段结束 =====
        
        # 添加用户消息到记忆
        self.memory.add_message(session_id, "user", query)
        
        # 初始化上下文
        context_data = {
            "query": query,
            "session_id": session_id,
            "retrieved_docs": [],
            "question_type": question_type,
            "is_safe": True,
            "steps": []
        }
        
        # 获取对话历史用于增强上下文
        history_context = self._get_history_context(session_id, query)
        
        # 步骤循环
        for step_idx in range(self.max_steps):
            logger.info(f"Agent 步骤 {step_idx + 1}/{self.max_steps}")
            
            # 早停检查：如果已有足够的检索文档，直接生成答案
            retrieved_docs = context_data.get("retrieved_docs", [])
            if len(retrieved_docs) >= self.config.min_docs_for_answer:
                logger.info(f"已检索到 {len(retrieved_docs)} 个文档，满足回答条件，提前终止")
                break
            
            # 1. 思考 + 决定行动 + 反思（合并为一次 LLM 调用）
            should_continue, thought, action, action_input = await self._think_decide_reflect(
                query, context_data, history_context, step_idx
            )
            
            self._add_step(thought=thought, state=AgentState.THINKING)
            
            if action is None or not should_continue:
                # 无需行动或已足够，跳出循环
                break
            
            # 2. 执行行动
            self.state = AgentState.ACTING
            observation = await self._execute_action(action, action_input, context_data)
            
            self._add_step(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                state=AgentState.ACTING
            )
            
            # 3. 更新上下文
            context_data = self._update_context(action, observation, context_data)
        
        # 生成最终答案
        answer, sources = await self._generate_final_answer(query, context_data)
        
        # 计算置信度
        confidence = self._calculate_confidence(context_data, query)
        
        # 生成追问
        followup_questions = []
        requires_followup = False
        if self.config.enable_followup:
            followup_result = await self._generate_followup(query, context_data)
            if followup_result:
                followup_questions = followup_result
                requires_followup = len(followup_questions) > 0
        
        # 识别知识缺口
        knowledge_gaps = []
        if self.config.enable_knowledge_gap and confidence < self.config.min_confidence_threshold:
            gap_result = await self._identify_knowledge_gap(query, context_data, confidence)
            if gap_result:
                knowledge_gaps = gap_result
        
        # 添加助手消息到记忆
        self.memory.add_message(
            session_id, 
            "assistant", 
            answer,
            metadata={
                "sources": sources,
                "confidence": confidence
            }
        )
        
        return AgentResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            disclaimer="本回答仅供参考，不能替代专业医疗建议。如有严重症状，请及时就医。",
            steps=self.steps,
            followup_questions=followup_questions,
            knowledge_gaps=knowledge_gaps,
            requires_followup=requires_followup
        )

    def _get_history_context(self, session_id: str, current_query: str) -> str:
        """获取对话历史上下文
        
        Args:
            session_id: 会话ID
            current_query: 当前问题
            
        Returns:
            str: 格式化的历史上下文
        """
        try:
            history = self.memory.get_conversation(session_id, count=3)
            if not history:
                return ""
            
            context_parts = []
            for msg in history[-3:]:  # 最近3轮
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]
                context_parts.append(f"{role}: {content}")
            
            return "\n".join(context_parts)
        except Exception:
            return ""

    def _generate_friendly_response(self, query: str, question_type: Optional[str]) -> str:
        """生成友好回答（非医疗问题）
        
        Args:
            query: 用户查询
            question_type: 问题类型
            
        Returns:
            str: 友好回答
        """
        query_lower = query.lower().strip()
        
        # 问候语回复
        if question_type == "greeting":
            greetings = [
                "您好！👋 我是医疗知识问答助手，专注于为您提供医疗健康相关的咨询帮助。请问有什么可以帮您的吗？",
                "您好呀！😊 我是您的医疗健康助手。如果您有任何关于疾病、症状、用药或检查报告的问题，欢迎随时向我咨询。",
                "嗨！✨ 我是医疗问答助手，擅长解答各种医疗健康问题。有什么需要帮助的吗？",
            ]
            return greetings[hash(query_lower) % len(greetings)]
        
        # 非医疗话题回复
        elif question_type == "off_topic":
            return f"抱歉，我是一个医疗知识问答助手，主要帮助您解答关于疾病、症状、用药、检查等方面的医疗问题。\n\n您的问题是「{query}」，这个话题我不太擅长。如果您有医疗健康方面的问题，欢迎随时问我！🏥"
        
        # 未知类型但非医疗
        return f"您好！我是一个专注于医疗健康领域的问答系统。您的问题是「{query}」，这个内容超出了我的专业范围。\n\n如果您有其他医疗健康方面的问题（如疾病症状、用药指导、检查结果解读等），我会很乐意帮助您！💊"

    async def _think_decide_reflect(
        self,
        query: str,
        context: Dict[str, Any],
        history_context: str,
        step_idx: int
    ) -> tuple[bool, str, Optional[str], Optional[Dict[str, Any]]]:
        """思考 + 决定行动 + 反思（合并为一次 LLM 调用）
        
        这是核心优化：减少 LLM 调用次数，从每步 2-3 次减少到 1 次
        
        Args:
            query: 用户查询
            context: 上下文
            history_context: 对话历史
            step_idx: 步骤索引
            
        Returns:
            tuple: (是否继续, 思考结果, 行动名, 行动输入)
        """
        docs = context.get("retrieved_docs", [])
        docs_summary = self._summarize_docs(docs)
        
        # 构建合并后的 prompt
        prompt = f"""你是一个医疗问答 Agent。请分析问题并决定下一步行动。

[步骤 {step_idx + 1}]

【用户问题】
{query}

【对话历史】
{history_context or "无历史对话"}

【当前状态】
- 已检索文档数: {len(docs)}
- 文档摘要: {docs_summary}
- 安全状态: {context.get("is_safe", True)}

【可执行行动】
1. check_security - 检查安全问题（敏感词、紧急症状）
2. retrieve_docs - 检索知识库文档
3. generate_followup_questions - 生成追问问题
4. identify_knowledge_gap - 识别知识缺口
5. done - 无需更多行动，可以生成答案

请按以下 JSON 格式输出：
{{
    "analysis": "你对问题的分析（1-2句话）",
    "action": "选择的行动名称",
    "action_input": {{"行动参数"}},
    "reason": "选择该行动的理由",
    "should_continue": true/false,
    "done_reason": "如果 action 是 done，说明为什么可以生成答案"
}}

要求：
- 如果 docs 数量 >= 2 且与问题相关，应选择 "done"
- 如果还没有检索文档，应先 "retrieve_docs"
- 如果需要更多上下文，可以 "generate_followup_questions"
- 如果置信度低，可以 "identify_knowledge_gap"
- always respond in JSON format"""
        
        try:
            response = self.rag_engine.llm_manager.generate(
                query=prompt,
                context="",
                question_type="reasoning"
            )
            
            # 解析 JSON 响应
            result = self._parse_json_response(response)
            
            if result:
                action = result.get("action")
                action_input = result.get("action_input", {})
                should_continue = result.get("should_continue", True)
                thought = f"{result.get('analysis', '')} → {result.get('reason', '')}"
                
                return should_continue, thought, action, action_input
            else:
                # 解析失败，使用默认逻辑
                return self._default_think_decide(query, context, step_idx)
                
        except Exception as e:
            logger.warning(f"合并思考调用LLM失败: {e}, 使用默认逻辑")
            return self._default_think_decide(query, context, step_idx)

    def _summarize_docs(self, docs: List[Dict[str, Any]]) -> str:
        """生成文档摘要
        
        Args:
            docs: 文档列表
            
        Returns:
            str: 摘要文本
        """
        if not docs:
            return "无"
        
        summaries = []
        for doc in docs[:3]:  # 最多3个
            text = doc.get("text", "")[:150]
            summaries.append(text)
        
        return " | ".join(summaries)

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析 JSON 响应
        
        Args:
            response: LLM 响应
            
        Returns:
            Optional[Dict]: 解析后的结果
        """
        import json
        import re
        
        if not response:
            return None
        
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 尝试直接解析
        try:
            return json.loads(response)
        except:
            return None

    def _default_think_decide(
        self,
        query: str,
        context: Dict[str, Any],
        step_idx: int
    ) -> tuple[bool, str, Optional[str], Optional[Dict[str, Any]]]:
        """默认的思考-决策逻辑（当 LLM 调用失败时使用）
        
        Args:
            query: 用户查询
            context: 上下文
            step_idx: 步骤索引
            
        Returns:
            tuple: (是否继续, 思考结果, 行动名, 行动输入)
        """
        docs = context.get("retrieved_docs", [])
        
        if not docs:
            # 还没有检索
            return True, "需要检索知识库", "retrieve_docs", {"query": query, "top_k": config.TOP_K}
        elif len(docs) >= 2:
            # 已有足够文档
            return False, "已获得足够信息", None, None
        else:
            # 文档较少，尝试补充
            return True, "检索结果较少，尝试补充检索", "retrieve_docs", {"query": query, "top_k": config.TOP_K}

    async def _think(self, query: str, context: Dict[str, Any], step_idx: int = 0) -> str:
        """思考步骤
        
        分析问题，决定下一步行动
        
        Args:
            query: 用户查询
            context: 上下文
            step_idx: 当前步骤索引（用于区分不同步骤的缓存）
            
        Returns:
            str: 思考结果
        """
        # 使用 LLM 进行思考，包含步骤索引使每次调用独特
        prompt = f"""[步骤 {step_idx + 1}] 请分析以下医疗问题，并决定下一步应该做什么：

用户问题: {query}

当前状态:
- 已检索文档数: {len(context.get('retrieved_docs', []))}
- 问题类型: {context.get('question_type', 'unknown')}
- 安全状态: {context.get('is_safe', True)}

请决定:
1. 是否需要检查安全问题?
2. 是否需要检索知识库?
3. 检索结果是否足够回答问题?
4. 是否需要生成追问问题?

请用中文回复，简洁明确。"""
        
        try:
            response = self.rag_engine.llm_manager.generate(
                query=prompt,
                context="",
                question_type="general"
            )
            return response if response else "分析问题中..."
        except Exception as e:
            logger.warning(f"思考步骤调用LLM失败: {e}")
            return "需要检索知识库来回答问题"

    def _decide_action(
        self,
        thought: str,
        context: Dict[str, Any]
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """决定行动
        
        从思考结果中解析出要执行的行动
        
        Args:
            thought: 思考结果
            context: 上下文
            
        Returns:
            tuple: (action, action_input)
        """
        thought_lower = thought.lower()
        
        # 检查是否需要安全检查
        if not context.get("is_safe", True) or "安全" in thought:
            return "check_security", {"query": context.get("query", "")}
        
        # 检查是否需要检索
        if "检索" in thought or "知识库" in thought or not context.get("retrieved_docs"):
            return "retrieve_docs", {
                "query": context.get("query", ""),
                "top_k": config.TOP_K
            }
        
        # 检查是否需要追问
        if "追问" in thought or "更多信息" in thought:
            return "generate_followup_questions", {
                "query": context.get("query", "")
            }
        
        # 检查是否需要识别知识缺口
        if "缺口" in thought or "不足" in thought:
            return "identify_knowledge_gap", {
                "query": context.get("query", ""),
                "confidence": 0.5
            }
        
        # 默认不需要更多行动
        return None, None

    async def _execute_action(
        self,
        action: str,
        action_input: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """执行行动
        
        调用相应的工具
        
        Args:
            action: 行动名称
            action_input: 行动输入
            context: 上下文
            
        Returns:
            str: 观察结果
        """
        tool = self.get_tool(action)
        if tool is None:
            return f"未找到工具: {action}"
        
        try:
            # 确保必要参数被传递（从context中获取）
            action_input = self._ensure_required_params(action, action_input, context)
            
            # 同步工具调用包装为异步
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tool(**action_input)
            )
            
            # 更新上下文
            context = self._update_context(action, result, context)
            
            return result
        except Exception as e:
            logger.error(f"工具执行失败: {action}, {e}")
            return f"工具执行失败: {str(e)}"

    def _ensure_required_params(
        self,
        action: str,
        action_input: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """确保工具调用包含必需参数
        
        Args:
            action: 行动名称
            action_input: 当前行动输入
            context: 上下文
            
        Returns:
            Dict: 补全后的行动输入
        """
        # 获取用户原始问题
        query = context.get("query", "")
        
        # 根据工具类型补充必要参数
        if action == "generate_followup_questions":
            if "query" not in action_input:
                action_input["query"] = query
            if "question_type" not in action_input:
                action_input["question_type"] = context.get("question_type", "unknown")
                
        elif action == "identify_knowledge_gap":
            if "query" not in action_input:
                action_input["query"] = query
            if "confidence" not in action_input:
                action_input["confidence"] = context.get("confidence", 0.5)
                
        elif action == "retrieve_docs":
            if "query" not in action_input:
                action_input["query"] = query
            if "top_k" not in action_input:
                action_input["top_k"] = config.TOP_K
                
        elif action == "check_security":
            if "query" not in action_input:
                action_input["query"] = query
        
        return action_input

    def _update_context(
        self,
        action: str,
        observation: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新上下文
        
        根据行动和观察结果更新上下文
        
        Args:
            action: 行动
            observation: 观察结果
            context: 当前上下文
            
        Returns:
            Dict: 更新后的上下文
        """
        if action == "retrieve_docs":
            try:
                result = json.loads(observation)
                if result.get("status") == "success":
                    context["retrieved_docs"] = result.get("documents", [])
            except:
                pass
        
        elif action == "check_security":
            try:
                result = json.loads(observation)
                context["is_safe"] = result.get("is_safe", True)
                if not result.get("is_safe"):
                    context["blocked_reason"] = result.get("blocked_reason")
            except:
                pass
        
        return context

    async def _reflect_with_observation(
        self,
        query: str,
        observation: str,
        context: Dict[str, Any]
    ) -> tuple[bool, str]:
        """反思步骤（带观察结果）
        
        检查当前结果是否足够回答问题
        
        Args:
            query: 用户查询
            observation: 观察结果
            context: 上下文
            
        Returns:
            tuple: (是否继续, 反思结果)
        """
        # 解析观察结果
        try:
            result = json.loads(observation) if observation.startswith("{") else {}
        except:
            result = {}
        
        # 检查是否有检索结果
        docs = context.get("retrieved_docs", [])
        
        if not docs:
            return True, "尚未检索到相关文档，需要继续检索"
        
        # 检查检索结果数量
        if len(docs) < 2:
            return True, "检索结果较少，考虑补充检索"
        
        # 检查观察结果是否表明需要继续
        if result.get("status") == "error":
            return False, f"工具执行出错: {result.get('message')}"
        
        # 默认可以结束
        return False, "已获得足够信息，可以生成回答"

    async def _reflect(self, query: str, context: Dict[str, Any]) -> str:
        """反思步骤（简化版）
        
        Args:
            query: 用户查询
            context: 上下文
            
        Returns:
            str: 反思结果
        """
        docs = context.get("retrieved_docs", [])
        
        if not docs:
            return "需要检索更多文档"
        
        return "已获得足够信息"

    async def _act(self, thought: str, context: Dict[str, Any]) -> str:
        """行动步骤
        
        根据思考结果执行相应的工具调用
        
        Args:
            thought: 思考结果，包含要执行的行动
            context: 当前上下文
            
        Returns:
            str: 行动观察结果
        """
        # 从思考结果中解析行动
        action, action_input = self._decide_action(thought, context)
        
        if action is None:
            return "无需执行更多行动"
        
        # 执行行动
        return await self._execute_action(action, action_input or {}, context)

    async def _generate_final_answer(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """生成最终回答
        
        Args:
            query: 用户查询
            context: 上下文
            
        Returns:
            tuple: (回答, 来源列表)
        """
        docs = context.get("retrieved_docs", [])
        
        if not docs:
            return "抱歉，知识库中未找到相关信息。请先上传医疗文档到知识库。", []
        
        # 提取上下文
        context_text = "\n\n".join([
            doc.get("text", "")[:1000] for doc in docs
        ])
        
        # 构建完整 prompt
        full_prompt = self._build_answer_prompt(query, context_text)
        
        # 生成回答
        try:
            answer = self.rag_engine.llm_manager.generate(
                query=full_prompt,
                context=context_text,
                question_type=context.get("question_type", "unknown")
            )
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            answer = "生成回答时出错，请稍后重试。"
        
        # 提取来源
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.get("text", "")[:200] + "...",
                "source": doc.get("metadata", {}).get("file_name", "未知"),
                "score": doc.get("score")
            })
        
        return answer, sources

    def _build_answer_prompt(self, query: str, context: str) -> str:
        """构建回答 prompt"""
        return f"""基于以下医疗知识库内容，请回答用户的问题。

要求：
1. 只根据提供的知识库内容回答，不要编造信息
2. 如果知识库中没有相关信息，请明确说明
3. 给出专业、准确、易懂的解释
4. 始终提供免责声明

知识库内容：
{context}

用户问题：{query}

请给出回答："""

    def _calculate_confidence(self, context: Dict[str, Any], query: str = "") -> float:
        """计算置信度（多维度加权）
        
        改进后的置信度计算：
        - 文档数量: 0-0.2
        - 相似度分数: 0-0.4
        - 问题覆盖度: 0-0.4
        
        Args:
            context: 上下文
            query: 用户问题（用于评估覆盖度）
            
        Returns:
            float: 置信度 (0-1)
        """
        docs = context.get("retrieved_docs", [])
        
        if not docs:
            return 0.0
        
        score = 0.0
        
        # 1. 文档数量得分 (0-0.2)
        # 0个=0, 1个=0.05, 2个=0.1, 3个=0.15, 4+=0.2
        count_score = min(len(docs) * 0.05, 0.2)
        score += count_score
        
        # 2. 平均相似度得分 (0-0.4)
        scores = [d.get("score", 0) for d in docs if d.get("score")]
        if scores:
            avg_score = sum(scores) / len(scores)
            # 相似度分数通常在 0-1 范围
            similarity_score = avg_score * 0.4
            score += similarity_score
        
        # 3. 文档与问题的相关度估算 (0-0.4)
        # 简单估算：检查文档是否包含问题的关键词
        if query:
            relevance_score = self._estimate_relevance(query, docs)
            score += relevance_score * 0.4
        
        # 4. 额外加分：有来源标注
        has_sources = any(d.get("metadata", {}).get("file_name") for d in docs)
        if has_sources:
            score += 0.1
        
        # 确保在 0-1 范围内
        return min(score, 1.0)

    def _estimate_relevance(self, query: str, docs: List[Dict[str, Any]]) -> float:
        """估算文档与问题的相关度
        
        Args:
            query: 用户问题
            docs: 文档列表
            
        Returns:
            float: 相关度 (0-1)
        """
        # 简单关键词匹配
        query_keywords = set(query.lower())
        # 去除常见停用词
        stopwords = {"的", "是", "在", "有", "和", "与", "或", "什么", "怎么", "如何", "为什么", "请问"}
        query_keywords = query_keywords - stopwords
        
        if not query_keywords:
            return 0.5  # 默认值
        
        relevant_count = 0
        for doc in docs:
            text = doc.get("text", "").lower()
            # 检查是否有任意关键词匹配
            if any(kw in text for kw in query_keywords):
                relevant_count += 1
        
        return relevant_count / len(docs) if docs else 0

    async def _generate_followup(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """生成追问问题
        
        Args:
            query: 用户查询
            context: 上下文
            
        Returns:
            List[str]: 追问问题列表
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.followup_tool.analyze_and_suggest(
                    query=query,
                    retrieved_docs=context.get("retrieved_docs", []),
                    question_type=context.get("question_type", "unknown")
                )
            )
            
            result_data = json.loads(result)
            return result_data.get("followup_questions", [])
            
        except Exception as e:
            logger.error(f"生成追问失败: {e}")
            return []

    async def _identify_knowledge_gap(
        self,
        query: str,
        context: Dict[str, Any],
        confidence: float
    ) -> List[str]:
        """识别知识缺口
        
        Args:
            query: 用户查询
            context: 上下文
            confidence: 置信度
            
        Returns:
            List[str]: 缺口描述列表
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.knowledge_gap_tool.identify_gaps(
                    query=query,
                    retrieved_docs=context.get("retrieved_docs", []),
                    confidence=confidence
                )
            )
            
            result_data = json.loads(result)
            gaps = result_data.get("gaps", [])
            return [g.get("description", "") for g in gaps]
            
        except Exception as e:
            logger.error(f"识别知识缺口失败: {e}")
            return []

    # 工具函数封装
    def _tool_retrieve(self, query: str, top_k: int = None, **kwargs) -> str:
        """检索工具封装"""
        return self.retriever_tool.retrieve(query, top_k=top_k or config.TOP_K, **kwargs)

    def _tool_security_check(self, query: str, **kwargs) -> str:
        """安全检查工具封装"""
        return self.security_tool.check(query)

    def _tool_generate_followup(self, query: str, **kwargs) -> str:
        """追问工具封装"""
        return self.followup_tool.generate_questions(query, **kwargs)

    def _tool_identify_gap(self, query: str, confidence: float = 0.5, **kwargs) -> str:
        """知识缺口工具封装"""
        return self.knowledge_gap_tool.identify_gaps(
            query=query,
            retrieved_docs=[],
            confidence=confidence
        )


def create_medical_agent(
    rag_engine,
    security_service,
    question_type_detector,
    confidence_calculator,
    config: Optional[AgentConfig] = None
) -> MedicalAgent:
    """创建医疗问答 Agent
    
    Args:
        rag_engine: RAG 引擎
        security_service: 安全检查服务
        question_type_detector: 问题类型检测器
        confidence_calculator: 置信度计算器
        config: Agent 配置
        
    Returns:
        MedicalAgent: 医疗问答 Agent 实例
    """
    return MedicalAgent(
        rag_engine=rag_engine,
        security_service=security_service,
        question_type_detector=question_type_detector,
        confidence_calculator=confidence_calculator,
        config=config
    )
