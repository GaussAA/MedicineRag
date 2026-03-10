"""主动追问工具

分析问题完整性，生成追问问题，引导用户提供更多信息。
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FollowUpAnalysis:
    """追问分析结果"""
    is_complete: bool
    missing_info: List[str]
    suggested_questions: List[str]
    reasoning: str


class FollowUpTool:
    """主动追问工具类"""

    def __init__(self, llm_manager=None):
        """初始化追问工具
        
        Args:
            llm_manager: LLM 管理器（用于生成追问）
        """
        self.llm_manager = llm_manager

    def analyze_and_suggest(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        question_type: str = "unknown"
    ) -> str:
        """分析问题并生成追问建议
        
        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档
            question_type: 问题类型
            
        Returns:
            str: 追问分析结果
        """
        try:
            # 分析问题完整性
            analysis = self._analyze_query_completeness(query, question_type)
            
            # 生成追问问题
            if not analysis.is_complete:
                followup_questions = self._generate_followup_questions(
                    query, analysis.missing_info, question_type, retrieved_docs
                )
            else:
                followup_questions = []
            
            result = {
                "status": "success",
                "query": query,
                "question_type": question_type,
                "is_complete": analysis.is_complete,
                "missing_info": analysis.missing_info,
                "followup_questions": followup_questions,
                "reasoning": analysis.reasoning,
                "requires_followup": len(followup_questions) > 0
            }
            
            logger.info(f"主动追问分析: query={query[:30]}..., complete={analysis.is_complete}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"追问分析异常: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            }, ensure_ascii=False)

    def generate_questions(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """生成追问问题（简化版）
        
        Args:
            query: 用户问题
            context: 额外上下文
            
        Returns:
            str: 追问问题列表
        """
        try:
            # 基于规则生成追问
            followup_questions = []
            
            # 检查症状描述完整性
            if any(kw in query for kw in ["症状", "不舒服", "疼", "痛"]):
                if "时间" not in query and "多久" not in query:
                    followup_questions.append("这个症状持续多长时间了？")
                if "伴随" not in query and "其他" not in query:
                    followup_questions.append("有没有伴随其他症状？")
            
            # 检查用药相关
            if any(kw in query for kw in ["药", "服用", "剂量"]):
                if "多久" not in query and "时间" not in query:
                    followup_questions.append("您服用这个药物多久了？")
                if "效果" not in query:
                    followup_questions.append("服药后效果如何？")
            
            # 检查检查相关
            if any(kw in query for kw in ["检查", "指标", "报告"]):
                if "什么时候" not in query and "时间" not in query:
                    followup_questions.append("这个检查是什么时候做的？")
            
            result = {
                "status": "success",
                "original_query": query,
                "followup_questions": followup_questions[:3],  # 最多3个
                "has_questions": len(followup_questions) > 0
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"生成追问异常: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            }, ensure_ascii=False)

    def _analyze_query_completeness(
        self,
        query: str,
        question_type: str
    ) -> FollowUpAnalysis:
        """分析问题完整性
        
        Args:
            query: 用户问题
            question_type: 问题类型
            
        Returns:
            FollowUpAnalysis: 分析结果
        """
        missing_info = []
        is_complete = True
        
        # 症状类问题检查
        if question_type == "symptom" or any(kw in query for kw in ["症状", "不舒服"]):
            if not any(kw in query for kw in ["多久", "时间", "持续"]):
                missing_info.append("症状持续时间")
                is_complete = False
            if not any(kw in query for kw in ["伴随", "其他", "还有"]):
                missing_info.append("伴随症状")
        
        # 用药类问题检查
        elif question_type == "medication" or any(kw in query for kw in ["药", "服用", "剂量"]):
            if not any(kw in query for kw in ["多久", "时间", "开始"]):
                missing_info.append("用药时长")
        
        # 疾病类问题检查
        elif question_type == "disease" or any(kw in query for kw in ["疾病", "确诊", "治疗"]):
            if not any(kw in query for kw in ["多久", "时间"]):
                missing_info.append("患病时长")
        
        # 检查类问题检查
        elif question_type == "examination" or any(kw in query for kw in ["检查", "指标"]):
            if not any(kw in query for kw in ["什么时候", "时间", "最近"]):
                missing_info.append("检查时间")
        
        reasoning = self._build_reasoning(is_complete, missing_info, question_type)
        
        return FollowUpAnalysis(
            is_complete=is_complete,
            missing_info=missing_info,
            suggested_questions=[],
            reasoning=reasoning
        )

    def _generate_followup_questions(
        self,
        query: str,
        missing_info: List[str],
        question_type: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[str]:
        """生成追问问题
        
        Args:
            query: 用户问题
            missing_info: 缺失信息列表
            question_type: 问题类型
            retrieved_docs: 检索到的文档
            
        Returns:
            List[str]: 追问问题列表
        """
        questions = []
        
        for info in missing_info:
            if info == "症状持续时间":
                questions.append("请问这个症状持续多长时间了？")
            elif info == "伴随症状":
                questions.append("有没有伴随其他不舒服的症状？")
            elif info == "用药时长":
                questions.append("您服用这个药物已经多久了？")
            elif info == "患病时长":
                questions.append("这个疾病确诊多长时间了？")
            elif info == "检查时间":
                questions.append("您说的这个检查是什么时候做的？")
        
        # 添加基于检索结果的追问
        if retrieved_docs:
            # 可以根据检索到的具体内容生成更精准的追问
            pass
        
        # 最多返回3个问题
        return questions[:3]

    def _build_reasoning(
        self,
        is_complete: bool,
        missing_info: List[str],
        question_type: str
    ) -> str:
        """构建推理说明"""
        if is_complete:
            return f"问题信息完整，可以直接回答"
        
        missing_str = "、".join(missing_info)
        return f"问题缺少以下关键信息：{missing_str}。建议追问以获取完整信息。"

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 schema"""
        return {
            "name": "generate_followup_questions",
            "description": "分析用户问题的完整性，判断是否需要追问以获取更多信息。当问题描述不够完整（如缺少症状持续时间、伴随症状等）时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户问题"
                    },
                    "retrieved_docs": {
                        "type": "array",
                        "description": "检索到的文档（可选，用于更精准的追问）",
                        "items": {
                            "type": "object"
                        }
                    },
                    "question_type": {
                        "type": "string",
                        "description": "问题类型：symptom/disease/medication/examination",
                        "enum": ["symptom", "disease", "medication", "examination", "unknown"]
                    }
                },
                "required": ["query"]
            }
        }


def create_followup_tool(llm_manager=None) -> FollowUpTool:
    """创建追问工具
    
    Args:
        llm_manager: LLM 管理器
        
    Returns:
        FollowUpTool: 追问工具实例
    """
    return FollowUpTool(llm_manager)
