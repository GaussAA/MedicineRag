"""问题类型检测服务"""

from typing import Optional

from rag.prompts import QUESTION_TYPE_KEYWORDS
from backend.logging_config import get_logger

logger = get_logger(__name__)


class QuestionTypeDetector:
    """问题类型检测器
    
    根据问题中的关键词识别问题类型：
    - symptom: 症状相关
    - disease: 疾病相关
    - medication: 用药相关
    - examination: 检查相关
    """
    
    def __init__(self):
        self.type_keywords = QUESTION_TYPE_KEYWORDS
    
    def detect(self, question: str) -> Optional[str]:
        """检测问题类型
        
        Args:
            question: 用户问题
            
        Returns:
            问题类型或None
        """
        question_lower = question.lower()
        
        # 统计每种类型的关键词匹配数
        type_scores = {}
        for qtype, keywords in self.type_keywords.items():
            score = sum(1 for kw in keywords if kw in question_lower)
            type_scores[qtype] = score
        
        # 返回得分最高的类型
        max_score = max(type_scores.values())
        if max_score > 0:
            for qtype, score in type_scores.items():
                if score == max_score:
                    logger.info(f"检测到问题类型: {qtype} (匹配{score}个关键词)")
                    return qtype
        
        return None
    
    def get_type_keywords(self, question_type: str) -> list:
        """获取指定类型的所有关键词"""
        return self.type_keywords.get(question_type, [])


# 全局实例
_detector: Optional[QuestionTypeDetector] = None


def get_question_type_detector() -> QuestionTypeDetector:
    """获取问题类型检测器实例"""
    global _detector
    if _detector is None:
        _detector = QuestionTypeDetector()
    return _detector


def detect_question_type(question: str) -> Optional[str]:
    """便捷函数：检测问题类型"""
    return get_question_type_detector().detect(question)
