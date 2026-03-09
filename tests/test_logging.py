"""测试日志配置模块"""

import pytest
import logging
from backend.logging_config import setup_logging, get_logger


class TestLoggingConfig:
    """测试日志配置"""

    def test_setup_logging(self):
        """测试日志配置"""
        logger = setup_logging(log_level="DEBUG")
        assert logger is not None
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, tmp_path):
        """测试带文件的日志配置"""
        log_file = tmp_path / "test.log"
        logger = setup_logging(log_level="DEBUG", log_file=str(log_file))
        assert logger is not None
        # 文件处理器应该被添加
        assert len(logger.handlers) >= 2  # console + file

    def test_get_logger(self):
        """测试获取logger"""
        logger = get_logger("test")
        assert logger is not None
        assert logger.name == "medical_rag.test"

    def test_get_logger_different_names(self):
        """测试不同名称的logger"""
        logger1 = get_logger("test1")
        logger2 = get_logger("test2")
        assert logger1.name == "medical_rag.test1"
        assert logger2.name == "medical_rag.test2"
