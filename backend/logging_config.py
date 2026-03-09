"""统一日志配置模块"""

import logging
import re
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class SensitiveDataFormatter(logging.Formatter):
    """日志脱敏格式化器 - 自动过滤敏感信息"""
    
    # 敏感信息正则模式
    PATTERNS = [
        (re.compile(r'\d{17}[\dXx]'), '**********'),  # 身份证号
        (re.compile(r'1[3-9]\d{9}'), '***********'),  # 手机号
        (re.compile(r'\d{4}-\d{4}-\d{4}-\d{4}'), '****-****-****-****'),  # 银行卡
        (re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), '***@***.***'),  # 邮箱
    ]
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录并脱敏"""
        # 调用原始格式化
        message = super().format(record)
        
        # 对消息进行脱敏
        for pattern, replacement in self.PATTERNS:
            message = pattern.sub(replacement, message)
        
        # 对record.msg进行脱敏（如果它是字符串）
        if isinstance(record.msg, str):
            for pattern, replacement in self.PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)
        
        # 对args进行脱敏
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.PATTERNS:
                        arg = pattern.sub(replacement, arg)
                new_args.append(arg)
            record.args = tuple(new_args)
        
        return message


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    统一日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则只输出到控制台
        log_format: 自定义日志格式
    
    Returns:
        配置好的logger对象
    """
    # 默认日志格式
    default_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    log_format = log_format or default_format
    
    # 创建logger
    logger = logging.getLogger("medical_rag")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式化器（使用脱敏版本）
    formatter = SensitiveDataFormatter(log_format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用RotatingFileHandler，单个文件最大10MB，保留5个备份
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger
    
    Args:
        name: logger名称，通常使用__name__
    
    Returns:
        logger对象
    """
    return logging.getLogger(f"medical_rag.{name}")


# 默认日志配置
default_logger = setup_logging()
