"""测试安全服务模块"""

import pytest
from backend.services.security_service import SecurityService, CheckResult


class TestSecurityService:
    """测试SecurityService类"""

    @pytest.fixture
    def security_service(self):
        """创建安全服务实例"""
        return SecurityService()

    def test_check_content_safe(self, security_service):
        """测试正常内容通过检查"""
        result = security_service.check_content("高血压有什么症状？")
        assert result.is_safe is True
        assert result.category is None

    def test_check_content_sensitive_suicide(self, security_service):
        """测试自杀相关内容被检测"""
        result = security_service.check_content("我想自杀怎么办")
        assert result.is_safe is False
        assert result.category == "suicide"
        assert result.warning_message is not None

    def test_check_content_sensitive_self_harm(self, security_service):
        """测试自残相关内容被检测"""
        result = security_service.check_content("我想伤害自己")
        assert result.is_safe is False
        assert result.category == "self_harm"

    def test_check_content_sensitive_violence(self, security_service):
        """测试暴力相关内容被检测"""
        result = security_service.check_content("我想杀人")
        assert result.is_safe is False
        assert result.category == "violence"

    def test_check_content_sensitive_drugs(self, security_service):
        """测试毒品相关内容被检测"""
        result = security_service.check_content("哪里可以买到毒品")
        assert result.is_safe is False
        assert result.category == "drugs"

    def test_check_content_empty(self, security_service):
        """测试空内容"""
        result = security_service.check_content("")
        assert result.is_safe is True

    def test_check_content_none(self, security_service):
        """测试None内容"""
        result = security_service.check_content(None)
        assert result.is_safe is True

    def test_desensitize_phone(self, security_service):
        """测试手机号脱敏"""
        text = "我的手机号是13812345678"
        result = security_service.desensitize(text)
        assert "13812345678" not in result

    def test_desensitize_id_card(self, security_service):
        """测试身份证号脱敏"""
        text = "我的身份证号是110101199001011234"
        result = security_service.desensitize(text)
        assert "110101199001011234" not in result

    def test_is_emergency_symptom_chest_pain(self, security_service):
        """测试胸痛被识别为紧急症状"""
        result = security_service.is_emergency_symptom("突然胸痛怎么办")
        assert result is True

    def test_is_emergency_symptom_breathing_difficulty(self, security_service):
        """测试呼吸困难被识别为紧急症状"""
        result = security_service.is_emergency_symptom("呼吸困难怎么办")
        assert result is True

    def test_is_emergency_symptom_normal(self, security_service):
        """测试正常症状不被识别为紧急"""
        result = security_service.is_emergency_symptom("感冒了吃什么药")
        assert result is False

    def test_get_emergency_message(self, security_service):
        """测试紧急消息返回"""
        message = security_service.get_emergency_message()
        assert message is not None
        assert "120" in message or "急救" in message or "医院" in message


class TestCheckResult:
    """测试CheckResult数据类"""

    def test_check_result_safe(self):
        """测试安全结果"""
        result = CheckResult(is_safe=True)
        assert result.is_safe is True
        assert result.category is None
        assert result.warning_message is None

    def test_check_result_unsafe(self):
        """测试不安全结果"""
        result = CheckResult(
            is_safe=False,
            category="suicide",
            warning_message="请拨打心理援助热线"
        )
        assert result.is_safe is False
        assert result.category == "suicide"
        assert result.warning_message == "请拨打心理援助热线"
