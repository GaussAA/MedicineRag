"""置信度计算服务"""

from typing import List, Dict, Any, Optional, Tuple

from backend.logging_config import get_logger

logger = get_logger(__name__)


class ConfidenceCalculator:
    """置信度计算器
    
    根据检索结果计算回答的置信度：
    - high: 相似度 >= 75%
    - medium: 60% <= 相似度 < 75%
    - low: 相似度 < 60%
    """
    
    # 置信度阈值配置
    HIGH_THRESHOLD = 75.0
    MEDIUM_THRESHOLD = 60.0
    
    def __init__(
        self,
        high_threshold: float = 75.0,
        medium_threshold: float = 60.0
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
    
    def calculate(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """计算置信度等级和警告信息
        
        Args:
            retrieved_docs: 检索到的文档列表
            
        Returns:
            (置信度等级, 警告信息)
        """
        if not retrieved_docs:
            return "low", "⚠️ 知识库中未找到相关内容"
        
        # 获取最高相似度分数
        top_score = retrieved_docs[0].get('score')
        
        if top_score is None:
            return "normal", ""
        
        # 转换为相似度百分比
        similarity = (1 - top_score) * 100
        
        if similarity < self.medium_threshold:
            warning = f"⚠️ 知识库匹配度较低（{similarity:.0f}%），回答仅供参考"
            return "low", warning
        elif similarity < self.high_threshold:
            warning = f"ℹ️ 知识库匹配度一般（{similarity:.0f}%）"
            return "medium", warning
        else:
            return "high", ""
    
    def calculate_with_sources(
        self,
        retrieved_docs: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """计算置信度并返回完整结果
        
        Args:
            retrieved_docs: 检索到的文档列表
            include_sources: 是否包含来源信息
            
        Returns:
            包含置信度、警告和来源的字典
        """
        confidence_level, warning = self.calculate(retrieved_docs)
        
        result = {
            "confidence_level": confidence_level,
            "warning": warning
        }
        
        if include_sources and retrieved_docs:
            sources = []
            for doc in retrieved_docs:
                metadata = doc.get('metadata', {})
                sources.append({
                    "text": doc.get('text', '')[:200] + "..." if len(doc.get('text', '')) > 200 else doc.get('text', ''),
                    "file_path": metadata.get('file_path', ''),
                    "chunk_id": metadata.get('chunk_id', 0),
                    "score": doc.get('score', 0)
                })
            result["sources"] = sources
        
        return result


# 全局实例
_calculator: Optional[ConfidenceCalculator] = None


def get_confidence_calculator() -> ConfidenceCalculator:
    """获取置信度计算器实例"""
    global _calculator
    if _calculator is None:
        _calculator = ConfidenceCalculator()
    return _calculator
