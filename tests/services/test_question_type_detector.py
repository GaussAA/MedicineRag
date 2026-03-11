"""问题类型检测器测试"""

import pytest
from backend.services.question_type_detector import (
    QuestionTypeDetector,
    get_question_type_detector,
    detect_question_type
)


class TestQuestionTypeDetector:
    """问题类型检测器测试"""
    
    def setup_method(self):
        """每个测试方法前初始化"""
        self.detector = QuestionTypeDetector()
    
    def test_detect_symptom_type(self):
        """测试症状类型检测"""
        # 症状类问题
        result = self.detector.detect("头痛怎么办？")
        assert result == "symptom"
        
        result = self.detector.detect("咳嗽不停怎么办")
        assert result == "symptom"
        
        result = self.detector.detect("发烧需要吃什么药")
        assert result == "symptom"
    
    def test_detect_disease_type(self):
        """测试疾病类型检测"""
        # 疾病类问题
        result = self.detector.detect("高血压如何治疗？")
        assert result == "disease"
        
        result = self.detector.detect("糖尿病需要注意什么")
        assert result == "disease"
        
        result = self.detector.detect("冠心病的症状")
        assert result == "disease"
    
    def test_detect_medication_type(self):
        """测试用药类型检测"""
        # 用药类问题 - 使用包含"药"关键词的问题
        result = self.detector.detect("降压药有哪些副作用？")
        assert result == "medication"
        
        result = self.detector.detect("这个药怎么服用")
        assert result == "medication"
    
    def test_detect_examination_type(self):
        """测试检查类型检测"""
        # 检查类问题
        result = self.detector.detect("体检需要检查哪些项目？")
        assert result == "examination"
        
        result = self.detector.detect("血糖检查需要注意什么")
        assert result == "examination"
    
    def test_detect_none_for_generic(self):
        """测试通用问题返回None（不含关键词的问题）"""
        # 不含任何关键词的问题返回None
        result = self.detector.detect("这个问题怎么回答")
        assert result is None
    
    def test_detect_greeting(self):
        """测试问候语识别"""
        # 问候语
        result = self.detector.detect("你好")
        assert result == "greeting"
        
        result = self.detector.detect("谢谢")
        assert result == "greeting"
        
        result = self.detector.detect("hello")
        assert result == "greeting"
        
        result = self.detector.detect("早上好")
        assert result == "greeting"
    
    def test_detect_off_topic(self):
        """测试非医疗话题识别"""
        # 非医疗话题
        result = self.detector.detect("今天天气怎么样")
        assert result == "off_topic"
        
        result = self.detector.detect("股票行情")
        assert result == "off_topic"
        
        result = self.detector.detect("今晚足球比赛")
        assert result == "off_topic"
    
    def test_detect_with_multiple_keywords(self):
        """测试多关键词问题"""
        # 多关键词问题
        result = self.detector.detect("糖尿病患者头痛怎么办？")
        # 应该匹配到symptom或disease
    
    def test_get_type_keywords(self):
        """测试获取类型关键词"""
        keywords = self.detector.get_type_keywords("symptom")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
    
    def test_global_instance(self):
        """测试全局实例"""
        detector1 = get_question_type_detector()
        detector2 = get_question_type_detector()
        assert detector1 is detector2
    
    def test_convenience_function(self):
        """测试便捷函数"""
        result = detect_question_type("高血压怎么治疗")
        assert result == "disease"