"""问题类型检测服务"""

from typing import Optional

from rag.core.prompts import QUESTION_TYPE_KEYWORDS
from backend.logging_config import get_logger

logger = get_logger(__name__)


# 非医疗问题类型（这些类型不需要检索知识库）
NON_MEDICAL_TYPES = {"greeting", "off_topic"}


class QuestionTypeDetector:
    """问题类型检测器
    
    根据问题中的关键词识别问题类型：
    - greeting: 问候闲聊类（你好、谢谢、再见等）
    - off_topic: 非医疗话题（天气、游戏等无关内容）
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
        question_lower = question.lower().strip()
        
        # 检查是否为空或太短
        if not question_lower or len(question_lower) < 2:
            logger.info("问题太短，识别为闲聊")
            return "off_topic"
        
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
        
        # 没有匹配到任何类型，检查问题长度
        # 短问题（小于5个字符）视为闲聊
        if len(question_lower) < 5:
            logger.info(f"问题较短且无关键词匹配，识别为闲聊: {question}")
            return "off_topic"
        
        return None
    
    def is_medical_related(self, question: str) -> bool:
        """判断问题是否与医疗相关
        
        Args:
            question: 用户问题
            
        Returns:
            True 表示需要查资料，False 表示不需要
        """
        question_type = self.detect(question)
        
        # 非医疗问题类型不需要查资料
        if question_type in NON_MEDICAL_TYPES:
            return False
        
        # None 表示无法识别为常见医疗类型，但可能是医疗问题，需要查资料
        # 为了安全起见，返回 True 让 Agent 判断
        if question_type is None:
            return True
        
        return True
    
    def is_greeting(self, question: str) -> bool:
        """判断是否是问候类问题"""
        return self.detect(question) == "greeting"
    
    def is_off_topic(self, question: str) -> bool:
        """判断是否是非医疗话题"""
        return self.detect(question) == "off_topic"
    
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


def is_medical_related(question: str) -> bool:
    """便捷函数：判断问题是否与医疗相关"""
    return get_question_type_detector().is_medical_related(question)
