"""知识缺口识别工具

识别知识库的薄弱领域，提示需要补充的文档方向。
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import Counter
import os

from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class KnowledgeGap:
    """知识缺口"""
    category: str
    description: str
    severity: str  # high/medium/low
    suggested_topics: List[str] = field(default_factory=list)


@dataclass
class KnowledgeGapReport:
    """知识缺口报告"""
    timestamp: str
    total_unanswered: int
    question_categories: Dict[str, int]
    gaps: List[KnowledgeGap]
    recommendations: List[str]


class KnowledgeGapTool:
    """知识缺口识别工具类"""

    def __init__(self, stats_file: str = "data/qa_stats.json"):
        """初始化知识缺口工具
        
        Args:
            stats_file: 问答统计数据文件路径
        """
        self.stats_file = stats_file
        self._ensure_stats_file()

    def _ensure_stats_file(self):
        """确保统计文件存在"""
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
        if not os.path.exists(self.stats_file):
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_questions": 0,
                    "questions": [],
                    "daily_stats": {}
                }, f, ensure_ascii=False)

    def identify_gaps(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        confidence: float
    ) -> str:
        """识别知识缺口
        
        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档
            confidence: 答案置信度
            
        Returns:
            str: 知识缺口分析结果
        """
        try:
            gaps = []
            
            # 检查是否无检索结果
            if not retrieved_docs:
                gaps.append(KnowledgeGap(
                    category="no_documents",
                    description="知识库中未找到相关文档",
                    severity="high",
                    suggested_topics=[self._extract_topic_from_query(query)]
                ))
            
            # 检查低置信度
            elif confidence < 0.5:
                gaps.append(KnowledgeGap(
                    category="low_confidence",
                    description="检索结果置信度较低，可能需要更多相关文档",
                    severity="medium",
                    suggested_topics=[self._extract_topic_from_query(query)]
                ))
            
            # 检查检索结果数量
            elif len(retrieved_docs) < 2:
                gaps.append(KnowledgeGap(
                    category="insufficient_docs",
                    description="检索到的相关文档数量较少",
                    severity="low",
                    suggested_topics=[self._extract_topic_from_query(query)]
                ))
            
            # 生成推荐
            recommendations = self._generate_recommendations(gaps)
            
            result = {
                "status": "success",
                "query": query,
                "confidence": confidence,
                "gaps": [
                    {
                        "category": g.category,
                        "description": g.description,
                        "severity": g.severity,
                        "suggested_topics": g.suggested_topics
                    }
                    for g in gaps
                ],
                "recommendations": recommendations,
                "has_gaps": len(gaps) > 0
            }
            
            logger.info(f"知识缺口识别: query={query[:30]}..., gaps={len(gaps)}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"知识缺口识别异常: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            }, ensure_ascii=False)

    def generate_gap_report(self) -> str:
        """生成知识缺口报告
        
        分析历史问答数据，识别知识库的薄弱环节
        
        Returns:
            str: 缺口报告
        """
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            questions = stats.get("questions", [])
            
            if not questions:
                return json.dumps({
                    "status": "success",
                    "message": "暂无问答数据",
                    "gaps": []
                }, ensure_ascii=False)
            
            # 分析问题类型分布
            type_counter = Counter()
            low_confidence_count = 0
            
            for q in questions:
                if q.get("question_type"):
                    type_counter[q["question_type"]] += 1
                if q.get("confidence", 1.0) < 0.5:
                    low_confidence_count += 1
            
            # 识别缺口
            gaps = []
            
            # 找出最少被问到的问题类型
            all_types = ["symptom", "disease", "medication", "examination"]
            type_counts = {t: type_counter.get(t, 0) for t in all_types}
            min_type = min(type_counts, key=type_counts.get)
            
            if type_counts[min_type] < 5:
                gaps.append(KnowledgeGap(
                    category="low_coverage",
                    description=f"'{min_type}'类型问题覆盖较少",
                    severity="medium",
                    suggested_topics=[f"更多{min_type}相关的医学文档"]
                ))
            
            # 低置信度问题
            if low_confidence_count / len(questions) > 0.3:
                gaps.append(KnowledgeGap(
                    category="high_low_confidence",
                    description="大量问题置信度较低，知识库可能存在系统性不足",
                    severity="high",
                    suggested_topics=["扩展知识库内容", "提升文档质量"]
                ))
            
            recommendations = self._generate_recommendations(gaps)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(questions),
                "question_categories": dict(type_counts),
                "low_confidence_count": low_confidence_count,
                "low_confidence_ratio": low_confidence_count / len(questions),
                "gaps": [
                    {
                        "category": g.category,
                        "description": g.description,
                        "severity": g.severity,
                        "suggested_topics": g.suggested_topics
                    }
                    for g in gaps
                ],
                "recommendations": recommendations
            }
            
            return json.dumps(report, ensure_ascii=False, indent=2)
            
        except FileNotFoundError:
            return json.dumps({
                "status": "success",
                "message": "暂无统计数据",
                "gaps": []
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"生成缺口报告异常: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            }, ensure_ascii=False)

    def record_unanswered(self, query: str, question_type: str = "unknown") -> str:
        """记录无法回答的问题
        
        Args:
            query: 用户问题
            question_type: 问题类型
            
        Returns:
            str: 记录结果
        """
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            # 添加未回答问题
            stats["questions"].append({
                "question": query,
                "question_type": question_type,
                "answered": False,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.0
            })
            
            # 只保留最近1000条
            if len(stats["questions"]) > 1000:
                stats["questions"] = stats["questions"][-1000:]
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            return json.dumps({
                "status": "success",
                "message": "问题已记录"
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"记录未回答问题异常: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            }, ensure_ascii=False)

    def _extract_topic_from_query(self, query: str) -> str:
        """从查询中提取主题"""
        # 简单提取关键词
        medical_keywords = [
            "症状", "疾病", "治疗", "药物", "检查", "诊断",
            "预防", "原因", "费用", "医保", "手术", "康复"
        ]
        
        for keyword in medical_keywords:
            if keyword in query:
                return f"关于'{keyword}'的更多文档"
        
        return "相关主题的更多文档"

    def _generate_recommendations(self, gaps: List[KnowledgeGap]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        for gap in gaps:
            if gap.severity == "high":
                recommendations.append(f"【重要】建议优先补充: {gap.description}")
            elif gap.severity == "medium":
                recommendations.append(f"建议补充: {gap.description}")
        
        if not recommendations:
            recommendations.append("知识库覆盖良好，继续保持")
        
        return recommendations

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 schema"""
        return {
            "name": "identify_knowledge_gap",
            "description": "分析当前问答是否存在知识缺口，识别知识库的薄弱环节，并给出补充建议。当检索结果不足或置信度低时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户问题"
                    },
                    "retrieved_docs": {
                        "type": "array",
                        "description": "检索到的文档列表",
                        "items": {
                            "type": "object"
                        }
                    },
                    "confidence": {
                        "type": "number",
                        "description": "当前答案置信度（0-1）"
                    }
                },
                "required": ["query", "confidence"]
            }
        }


def create_knowledge_gap_tool(stats_file: str = "data/qa_stats.json") -> KnowledgeGapTool:
    """创建知识缺口工具
    
    Args:
        stats_file: 统计数据文件路径
        
    Returns:
        KnowledgeGapTool: 知识缺口工具实例
    """
    return KnowledgeGapTool(stats_file)
