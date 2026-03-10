"""置信度计算器测试"""

import pytest
from backend.services.confidence_calculator import (
    ConfidenceCalculator,
    get_confidence_calculator
)


class TestConfidenceCalculator:
    """置信度计算器测试"""
    
    def setup_method(self):
        """每个测试方法前初始化"""
        self.calculator = ConfidenceCalculator()
    
    def test_calculate_high_confidence(self):
        """测试高置信度 - 相似度 >= 75%"""
        docs = [
            {'score': 0.9},  # 90% 相似度
            {'score': 0.8},
        ]
        level, warning = self.calculator.calculate(docs)
        assert level == "high"
        assert "匹配度" in warning  # 现在高匹配度也显示
        assert "90%" in warning
    
    def test_calculate_medium_confidence(self):
        """测试中等置信度 - 60% <= 相似度 < 75%"""
        docs = [
            {'score': 0.7},  # 70% 相似度
            {'score': 0.6},
        ]
        level, warning = self.calculator.calculate(docs)
        assert level == "medium"
        assert "一般" in warning
    
    def test_calculate_low_confidence(self):
        """测试低置信度 - 相似度 < 60%"""
        docs = [
            {'score': 0.5},  # 50% 相似度
            {'score': 0.4},
        ]
        level, warning = self.calculator.calculate(docs)
        assert level == "low"
        assert "较低" in warning
    
    def test_calculate_empty_docs(self):
        """测试空文档列表"""
        docs = []
        level, warning = self.calculator.calculate(docs)
        assert level == "low"
        assert "未找到" in warning
    
    def test_calculate_no_score(self):
        """测试无分数的文档"""
        docs = [
            {'text': 'some text'},
        ]
        level, warning = self.calculator.calculate(docs)
        assert level == "normal"
        assert warning == ""
    
    def test_calculate_with_sources(self):
        """测试带来源的计算"""
        docs = [
            {
                'text': '这是文档内容',
                'score': 0.9,  # 90% 相似度
                'metadata': {'file_path': 'test.txt', 'chunk_id': 0}
            }
        ]
        result = self.calculator.calculate_with_sources(docs)
        
        assert result['confidence_level'] == 'high'
        assert "匹配度" in result['warning']  # 现在高匹配度也显示
        assert 'sources' in result
        assert len(result['sources']) == 1
        assert result['sources'][0]['file_path'] == 'test.txt'
    
    def test_calculate_without_sources(self):
        """测试不包含来源的计算"""
        docs = [
            {'score': 0.9}
        ]
        result = self.calculator.calculate_with_sources(docs, include_sources=False)
        
        assert 'sources' not in result
    
    def test_custom_thresholds(self):
        """测试自定义阈值"""
        calculator = ConfidenceCalculator(high_threshold=80.0, medium_threshold=50.0)
        
        # 70% - 应该返回medium因为低于80但高于50
        docs = [{'score': 0.7}]  # 70% 相似度
        level, _ = calculator.calculate(docs)
        assert level == "medium"
        
        # 85% - 应该返回high因为高于80
        docs = [{'score': 0.85}]  # 85% 相似度
        level, _ = calculator.calculate(docs)
        assert level == "high"
    
    def test_global_instance(self):
        """测试全局实例"""
        calc1 = get_confidence_calculator()
        calc2 = get_confidence_calculator()
        assert calc1 is calc2
