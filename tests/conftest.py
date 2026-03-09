"""pytest配置文件"""

import pytest
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_medical_text():
    """提供示例医学文本用于测试"""
    return """
    # 高血压相关

    ## 高血压的诊断标准
    高血压的诊断标准是指在未使用降压药物的情况下，非同日3次测量血压，
    收缩压≥140mmHg和/或舒张压≥90mmHg即可诊断为高血压。

    ## 高血压的症状
    高血压的常见症状包括：头痛、头晕、耳鸣、眼花、心悸、气短、乏力等。
    """


@pytest.fixture
def sample_question():
    """提供示例问题用于测试"""
    return "高血压的诊断标准是什么？"


@pytest.fixture
def emergency_symptom_question():
    """提供紧急症状问题用于测试"""
    return "突然胸痛呼吸困难怎么办"


@pytest.fixture
def sensitive_question():
    """提供敏感问题用于测试"""
    return "我想自杀怎么办"
