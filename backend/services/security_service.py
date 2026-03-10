"""安全服务模块 - 敏感词过滤和内容安全检查（扩展版）"""

import re
import os
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from rag.core.prompts import SENSITIVE_WARNING_MESSAGES
from backend.logging_config import get_logger

logger = get_logger(__name__)


# 敏感词库（扩展版 - 200+词汇）
SENSITIVE_KEYWORDS = {
    # 自杀相关
    "suicide": [
        "自杀", "自尽", "轻生", "寻死", "不想活了", "想死", "想不开",
        "割腕", "上吊", "跳楼", "服毒", "投河", "卧轨", "撞车",
        "结束生命", "结束自己", "死亡方法", "如何死", "怎么死",
        "自杀方法", "自杀技巧", "寻死觅活", "活够了", "死了算了",
        "氰化物", "安眠药过量", "百草枯", "老鼠药"
    ],
    # 自残相关
    "self_harm": [
        "自残", "伤害自己", "割自己", "打自己", "烫自己",
        "砍自己", "剁手", "自虐", "自我伤害", "故意受伤",
        "用刀割", "用火烧", "的头", "扇自己"
    ],
    # 暴力相关
    "violence": [
        "杀人", "杀掉", "弄死", "整死", "弄死他", "要他命",
        "伤害他人", "杀人犯", "行凶", "作案", "报复社会",
        "砍人", "捅人", "杀人犯法", "杀人偿命", "宰人",
        "弄残", "打断腿", "下毒", "投毒", "毒死"
    ],
    # 毒品相关
    "drugs": [
        "毒品", "吸毒", "贩毒", "制毒", "种毒", "运毒",
        "海洛因", "冰毒", "K粉", "摇头丸", "大麻", "可卡因",
        "鸦片", "吗啡", "杜冷丁", "美沙酮", "兴奋剂",
        "走私毒品", "贩卖毒品", "容留吸毒", "代购毒品"
    ],
    # 色情相关
    "porn": [
        "色情", "黄色", "裸聊", "性交易", "约炮", "一夜情",
        "成人网站", "黄色网站", "援交", "买春", "卖淫",
        "嫖娼", "脱衣舞", "人体艺术", "裸体", "露点"
    ],
    # 医疗广告违规（新增）
    "medical_ad": [
        "根治", "根治", "包治", "包好", "无效退款", "无效退款",
        "最好", "最强", "第一", "顶级", "国家级", "世界级",
        "祖传", "秘方", "偏方", "神医", "神药", "灵丹妙药",
        "莆田系", "民营医院", "男科医院", "妇科医院", "人流医院",
        "割包皮", "阳痿早泄", "性病", "梅毒", "淋病", "尖锐湿疣",
        "试管婴儿", "代孕", "胎儿性别", "生男生女", "清宫表",
        "堕胎", "打胎", "流产", "引产"
    ],
    # 医托相关（新增）
    "medical_scammer": [
        "医托", "号贩子", "专家号", "加号", "内部票",
        "认识主任", "认识院长", "帮忙安排", "加塞", "插队",
        "绿色通道", "快速住院", "床位", "病房", "手术安排"
    ],
    # 医疗纠纷相关（新增）
    "medical_dispute": [
        "误诊", "医疗事故", "医疗过错", "医疗损害", "手术失败",
        "医疗维权", "医疗赔偿", "医疗诉讼", "医闹", "伤医",
        "杀医", "打医生", "砍医生", "医暴力", "医疗暴力"
    ],
    # 非法行医相关（新增）
    "illegal_practice": [
        "无证行医", "非法行医", "黑诊所", "假医生", "假文凭",
        "职称买卖", "论文买卖", "挂靠", "租证", "借证",
        "非法美容", "非法整形", "黑美容", "微整形", "注射美容"
    ],
    # 金融诈骗相关（新增）
    "fraud": [
        "诈骗", "骗局", "传销", "非法集资", "资金盘",
        "跑路", "崩盘", "维权", "报案", "立案",
        "杀猪盘", "电信诈骗", "诈骗犯", "骗子", "诈骗集团"
    ],
}


@dataclass
class CheckResult:
    """内容检查结果"""
    is_safe: bool
    category: Optional[str] = None
    warning_message: Optional[str] = None


class SecurityService:
    """安全服务类 - 负责敏感词过滤和内容安全检查"""

    def __init__(self):
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict:
        """编译正则表达式模式"""
        patterns = {}
        for category, keywords in SENSITIVE_KEYWORDS.items():
            # 构建正则表达式，匹配任意一个关键词
            pattern = "|".join(re.escape(kw) for kw in keywords)
            patterns[category] = re.compile(pattern, re.IGNORECASE)
        return patterns

    def check_content(self, text: str) -> CheckResult:
        """检查内容是否包含敏感词

        Args:
            text: 待检查的文本

        Returns:
            CheckResult: 检查结果，包含是否安全、敏感类别和警告消息
        """
        if not text:
            return CheckResult(is_safe=True)

        for category, pattern in self.patterns.items():
            if pattern.search(text):
                warning_message = SENSITIVE_WARNING_MESSAGES.get(
                    category,
                    "无法回答此问题，请咨询其他健康相关话题。"
                )
                return CheckResult(
                    is_safe=False,
                    category=category,
                    warning_message=warning_message
                )

        return CheckResult(is_safe=True)

    def desensitize(self, text: str) -> str:
        """脱敏处理 - 去除个人身份信息

        Args:
            text: 待脱敏的文本

        Returns:
            str: 脱敏后的文本
        """
        if not text:
            return text

        # 去除身份证号 (18位)
        text = re.sub(r'\d{17}[\dXx]', '**********', text)
        # 去除手机号
        text = re.sub(r'1[3-9]\d{9}', '***********', text)
        # 去除姓名（简单处理，常见的单复姓）
        text = re.sub(r'[张李王刘陈杨赵黄周吴郑孙朱马胡林郭何高罗]{1,2}先生', '***先生', text)
        text = re.sub(r'[张李王刘陈杨赵黄周吴郑孙朱马胡林郭何高罗]{1,2}女士', '***女士', text)

        return text

    def is_emergency_symptom(self, text: str) -> bool:
        """检查是否为紧急症状

        Args:
            text: 待检查的文本

        Returns:
            bool: 是否为紧急症状
        """
        emergency_keywords = [
            # 原有关键词
            "胸痛", "呼吸困难", "心脏", "休克", "昏厥",
            "大出血", "中风", "脑溢血", "瘫痪", "严重过敏",
            # 心血管系统
            "急性胸痛", "压榨性胸痛", "濒死感", "心悸", "心跳过速",
            "心跳过慢", "胸闷", "心律失常", "心脏骤停", "心肌梗死",
            # 神经系统
            "突发偏瘫", "口眼歪斜", "言语不清", "肢体麻木",
            "意识障碍", "抽搐", "癫痫", "脑卒中",
            # 呼吸系统
            "呼吸急促", "咯血", "咳血", "气胸", "哮喘持续",
            # 消化系统
            "急性腹痛", "刀割样腹痛", "消化道出血", "呕血", "黑便",
            # 过敏
            "过敏性休克", "严重皮疹",
            # 中毒与创伤
            "药物中毒", "农药中毒", "一氧化碳中毒",
            # 其他紧急
            "高热不退", "酮症酸中毒"
        ]

        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in emergency_keywords)

    def get_emergency_message(self) -> str:
        """获取紧急症状提示消息"""
        return "⚠️ 警告：您描述的症状可能涉及紧急情况，请立即拨打120急救电话或前往最近医院就诊！"
