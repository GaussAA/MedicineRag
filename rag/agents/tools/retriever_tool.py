"""文档检索工具

封装 RAG 引擎的检索功能为 Agent 可调用的工具。
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from backend.logging_config import get_logger
from backend.exceptions import VectorStoreError

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    documents: List[Dict[str, Any]]
    total: int
    query: str
    search_type: str = "hybrid"


class RetrieverTool:
    """文档检索工具类"""

    def __init__(self, rag_engine):
        """初始化检索工具
        
        Args:
            rag_engine: RAG 引擎实例
        """
        self.rag_engine = rag_engine

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hybrid: bool = True,
        search_type: str = "auto"
    ) -> str:
        """检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_hybrid: 是否使用混合检索
            search_type: 搜索类型 (auto/symptom/disease/medication/examination)
            
        Returns:
            str: 检索结果 JSON 字符串
        """
        try:
            # 根据搜索类型调整查询
            adjusted_query = self._adjust_query_by_type(query, search_type)
            
            # 执行检索
            docs = self.rag_engine.retrieve(
                query=adjusted_query,
                top_k=top_k or 5,
                use_hybrid=use_hybrid
            )
            
            # 格式化结果
            result = {
                "status": "success",
                "query": query,
                "adjusted_query": adjusted_query,
                "total": len(docs),
                "documents": []
            }
            
            for i, doc in enumerate(docs):
                doc_info = {
                    "index": i + 1,
                    "text": doc.get("text", "")[:500],  # 限制长度
                    "score": doc.get("score"),
                    "metadata": doc.get("metadata", {})
                }
                result["documents"].append(doc_info)
                
            logger.info(f"检索成功: query={query}, total={len(docs)}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except VectorStoreError as e:
            logger.error(f"检索失败: {e}")
            return json.dumps({
                "status": "error",
                "message": f"检索失败: {str(e)}"
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"检索异常: {e}")
            return json.dumps({
                "status": "error",
                "message": f"检索异常: {str(e)}"
            }, ensure_ascii=False)

    def retrieve_with_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> str:
        """基于上下文的检索（多轮对话）
        
        Args:
            query: 当前查询
            context: 对话历史上下文
            top_k: 返回结果数量
            
        Returns:
            str: 检索结果
        """
        # 构建增强查询
        enhanced_query = self._build_enhanced_query(query, context)
        return self.retrieve(enhanced_query, top_k=top_k)

    def _adjust_query_by_type(self, query: str, search_type: str) -> str:
        """根据搜索类型调整查询"""
        if search_type == "auto" or search_type == "symptom":
            return query
        elif search_type == "disease":
            return f"{query} 疾病 诊断 治疗"
        elif search_type == "medication":
            return f"{query} 药物 药品 用法 剂量"
        elif search_type == "examination":
            return f"{query} 检查 检验 指标"
        return query

    def _build_enhanced_query(self, query: str, context: List[Dict[str, Any]]) -> str:
        """构建增强查询（融入历史上下文）"""
        if not context:
            return query
            
        # 提取历史问题
        history_queries = []
        for item in context[-3:]:  # 只取最近3轮
            if item.get("role") == "user":
                history_queries.append(item.get("content", ""))
        
        if history_queries:
            history_text = " | ".join(history_queries[-3:])
            return f"{history_text} | {query}"
        
        return query

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 schema"""
        return {
            "name": "retrieve_docs",
            "description": "从医疗知识库中检索相关文档。当需要回答医学问题时，必须先使用此工具获取参考资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户问题或搜索关键词"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认5",
                        "default": 5
                    },
                    "use_hybrid": {
                        "type": "boolean",
                        "description": "是否使用混合检索（向量+关键词）",
                        "default": True
                    },
                    "search_type": {
                        "type": "string",
                        "description": "搜索类型：auto/symptom/disease/medication/examination",
                        "enum": ["auto", "symptom", "disease", "medication", "examination"],
                        "default": "auto"
                    }
                },
                "required": ["query"]
            }
        }


def create_retriever_tool(rag_engine) -> RetrieverTool:
    """创建检索工具
    
    Args:
        rag_engine: RAG 引擎实例
        
    Returns:
        RetrieverTool: 检索工具实例
    """
    return RetrieverTool(rag_engine)
