"""LLM管理模块

封装Ollama LLM的所有调用逻辑，
提供同步/流式生成接口。
"""

import time
from typing import List, Dict, Any, Optional, Iterator, Generator

from ollama import Client as OllamaClient

from backend.config import config
from backend.logging_config import get_logger
from backend.exceptions import LLMException, LLMTTimeoutError
from rag.core.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, QUESTION_TYPE_PROMPTS
from rag.cache import LLMResponseCache

logger = get_logger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """重试装饰器（指数退避）"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {delay}s: {e}"
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after max retries: {e}")
            raise last_exception
        return wrapper
    return decorator


class LLMManager:
    """LLM管理器 - 封装Ollama LLM调用"""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        cache: Optional[LLMResponseCache] = None
    ):
        """初始化LLM管理器

        Args:
            client: Ollama客户端
            cache: LLM响应缓存
        """
        # 创建客户端
        if client is None:
            self._client = OllamaClient(host=config.OLLAMA_BASE_URL)
        else:
            self._client = client

        # 初始化缓存
        self._cache = cache or LLMResponseCache(max_size=50, enabled=True)

    @property
    def client(self) -> OllamaClient:
        """获取Ollama客户端"""
        return self._client

    @property
    def cache(self) -> LLMResponseCache:
        """获取LLM响应缓存"""
        return self._cache

    def generate(
        self,
        query: str,
        context: str,
        question_type: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """同步生成回答

        Args:
            query: 用户问题
            context: 检索到的上下文
            question_type: 问题类型
            use_cache: 是否使用缓存

        Returns:
            生成的文本

        Raises:
            LLMException: LLM调用失败
        """
        # 构建prompt
        full_prompt = self._build_prompt(query, context, question_type)

        # 尝试从缓存获取
        if use_cache:
            # 构造虚拟的retrieved_docs用于缓存键生成
            mock_docs = [{"text": context[:500]}]
            cached_response = self._cache.get(query, mock_docs, question_type)
            if cached_response:
                logger.info("使用缓存的LLM响应")
                return cached_response

        # 调用LLM
        try:
            response = self._client.chat(
                model=config.OLLAMA_LLM_MODEL,
                messages=[{'role': 'user', 'content': full_prompt}],
                options={
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS
                }
            )

            result = response.message.content
            logger.info("LLM生成完成")

            # 缓存响应
            if use_cache:
                self._cache.put(query, [{"text": context[:500]}], result, question_type)

            return result

        except Exception as e:
            logger.error(f"LLM生成失败: {type(e).__name__}: {e}")
            raise LLMException(f"LLM生成失败: {str(e)}") from e

    def generate_stream(
        self,
        query: str,
        context: str,
        question_type: Optional[str] = None,
        full_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """流式生成回答

        Args:
            query: 用户问题
            context: 检索到的上下文
            question_type: 问题类型
            full_prompt: 预先构建的完整prompt

        Yields:
            逐字生成的回答

        Raises:
            LLMException: LLM调用失败
        """
        # 如果提供了完整prompt，直接使用
        if full_prompt:
            prompt_to_use = full_prompt
        else:
            prompt_to_use = self._build_prompt(query, context, question_type)

        try:
            logger.debug("开始调用LLM流式生成...")

            response = self._client.chat(
                model=config.OLLAMA_LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt_to_use}],
                options={
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS
                },
                stream=True
            )

            # 逐字yield
            for chunk in response:
                if chunk.message and chunk.message.content:
                    yield chunk.message.content

            logger.info("LLM流式生成完成")

        except Exception as e:
            logger.error(f"LLM流式生成失败: {type(e).__name__}: {e}")
            raise LLMException(f"流式生成失败: {str(e)}") from e

    def _build_prompt(
        self,
        query: str,
        context: str,
        question_type: Optional[str] = None
    ) -> str:
        """构建完整的prompt

        Args:
            query: 用户问题
            context: 检索到的上下文
            question_type: 问题类型

        Returns:
            完整的prompt
        """
        # 选择合适的prompt模板
        if question_type and question_type in ["symptom", "disease", "medication", "examination"]:
            type_prompt = QUESTION_TYPE_PROMPTS.get(question_type, "")
            context_prompt = f"{context}\n\n{type_prompt}"
        else:
            context_prompt = context

        # 构造完整prompt
        return f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(context=context_prompt, question=query)}"

    def build_prompt_with_history(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """构建含历史上下文的prompt

        Args:
            query: 当前问题
            history: 对话历史

        Returns:
            包含历史上下文的prompt
        """
        if not history:
            return query

        # 配置参数
        max_history_turns = 5
        max_answer_length = 300

        # 保留最近N轮对话
        recent_history = history[-max_history_turns:] if len(history) > max_history_turns else history

        # 构建历史上下文（带长度控制）
        context_parts = []
        total_length = 0
        max_context_length = 1500

        for h in reversed(recent_history):
            question = h.get('question', '')[:200]
            answer = h.get('answer', '')[:max_answer_length]

            turn = f"用户：{question}\n助手：{answer}"
            if total_length + len(turn) > max_context_length:
                break
            context_parts.insert(0, turn)
            total_length += len(turn)

        if not context_parts:
            return query

        context = "\n\n".join(context_parts)

        return f"""对话历史：
{context}

当前问题：{query}

请根据以上对话历史和当前问题进行回答。如果当前问题是对之前问题的追问或补充，请结合上下文回答。"""

    def generate_with_retry(
        self,
        query: str,
        context: str,
        question_type: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """带重试的生成（指数退避）

        Args:
            query: 用户问题
            context: 检索到的上下文
            question_type: 问题类型
            max_retries: 最大重试次数

        Returns:
            生成的文本

        Raises:
            LLMException: 所有重试都失败
        """
        @retry_with_backoff(max_retries=max_retries, initial_delay=2.0, backoff_factor=2.0)
        def _generate():
            return self.generate(query, context, question_type, use_cache=False)

        return _generate()

    def generate_fallback(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """生成fallback回答（当LLM失败时的降级方案）

        Args:
            query: 用户问题
            retrieved_docs: 检索到的文档

        Returns:
            基于检索资料的fallback回答
        """
        if not retrieved_docs:
            return "抱歉，知识库中未找到相关信息。请先上传医疗文档到知识库。"

        # 构建参考来源列表
        sources = []
        for i, doc in enumerate(retrieved_docs[:3]):
            text = doc.get('text', '')[:200]
            source = doc.get('metadata', {}).get('source', '未知来源')
            sources.append(f"{i+1}. {text}... (来源: {source})")

        sources_text = "\n\n".join(sources)

        return f"""抱歉，AI生成回答失败。以下是检索到的相关资料，请参考：

{sources_text}

温馨提示：建议稍后重试，如果问题持续存在请联系管理员。"""

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        logger.info("LLM响应缓存已清空")

    def is_available(self) -> bool:
        """检查LLM是否可用"""
        try:
            # 尝试调用模型列表
            self._client.list()
            return True
        except Exception as e:
            logger.warning(f"LLM不可用: {e}")
            return False


def create_llm_manager(
    client: Optional[OllamaClient] = None,
    cache: Optional[LLMResponseCache] = None
) -> LLMManager:
    """创建LLM管理器的工厂函数

    Args:
        client: Ollama客户端
        cache: LLM响应缓存

    Returns:
        LLMManager 实例
    """
    return LLMManager(client=client, cache=cache)


# 导出
__all__ = [
    'LLMManager',
    'retry_with_backoff',
    'create_llm_manager',
]
