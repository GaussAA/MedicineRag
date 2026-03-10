'''文档处理模块

封装文档解析和分块逻辑，
支持多种文档格式的智能处理。
'''

import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag.processing.chunker import IntelligentChunker
from rag.cache import ChunkCache
from backend.config import config
from backend.logging_config import get_logger
from backend.exceptions import DocumentParseError, ChunkingError

logger = get_logger(__name__)


class DocumentProcessor:
    """文档处理器 - 负责文档解析和分块"""

    def __init__(
        self,
        chunker: Optional[IntelligentChunker] = None,
        chunk_cache: Optional[ChunkCache] = None
    ):
        """初始化文档处理器

        Args:
            chunker: 分块器实例
            chunk_cache: 分块缓存实例
        """
        self._chunker = chunker or IntelligentChunker(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        self._chunk_cache = chunk_cache or ChunkCache(max_size=20)

    @property
    def chunker(self) -> IntelligentChunker:
        """获取分块器"""
        return self._chunker

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """处理单个文件

        Args:
            file_path: 文件路径

        Returns:
            处理结果字典

        Raises:
            DocumentParseError: 文档解析失败
            ChunkingError: 分块失败
        """
        try:
            # 读取文档内容
            content = self._read_file(file_path)

            if not content.strip():
                raise DocumentParseError(f"文档内容为空: {file_path}")

            # 生成分块
            chunks = self._chunk_text(content, file_path)

            logger.info(f"成功处理文件 {Path(file_path).name}: {len(chunks)} 个块")

            return {
                "status": "success",
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "chunk_count": len(chunks),
                "chunks": chunks
            }

        except DocumentParseError:
            raise
        except ChunkingError:
            raise
        except Exception as e:
            logger.error(f"处理文件失败: {file_path}, error: {e}")
            raise DocumentParseError(f"处理文件失败: {str(e)}") from e

    def _read_file(self, file_path: str) -> str:
        """读取文件内容

        Args:
            file_path: 文件路径

        Returns:
            文档文本内容

        Raises:
            DocumentParseError: 文档解析失败
        """
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()
        content = ""

        try:
            if suffix == '.txt':
                content = self._read_txt(file_path)
            elif suffix == '.pdf':
                content = self._read_pdf(file_path)
            elif suffix in ['.docx', '.doc']:
                content = self._read_docx(file_path)
            elif suffix == '.md':
                content = self._read_md(file_path)
            elif suffix == '.html':
                content = self._read_html(file_path)
            else:
                # 尝试使用通用解析器
                content = self._read_with_llama_index(file_path)

            logger.info(f"读取文档 {file_path_obj.name}: {len(content)} 字符")
            return content

        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, error: {e}")
            raise DocumentParseError(f"读取文件失败: {str(e)}") from e

    def _read_txt(self, file_path: str) -> str:
        """读取文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_pdf(self, file_path: str) -> str:
        """读取PDF文件"""
        # 优先使用 pymupdf4llm（高精度，支持Markdown）
        try:
            import pymupdf4llm
            md_text = pymupdf4llm.to_markdown(file_path)
            logger.info("使用 pymupdf4llm 解析 PDF")
            return md_text
        except ImportError:
            logger.warning("pymupdf4llm 未安装，回退到 SimpleDirectoryReader")
            return self._read_with_llama_index(file_path)
        except Exception as e:
            logger.warning(f"pymupdf4llm 解析失败: {e}，回退到 SimpleDirectoryReader")
            return self._read_with_llama_index(file_path)

    def _read_docx(self, file_path: str) -> str:
        """读取Word文件"""
        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs]
            return '\n\n'.join(paragraphs)
        except ImportError:
            logger.warning("python-docx 未安装，回退到 SimpleDirectoryReader")
            return self._read_with_llama_index(file_path)

    def _read_md(self, file_path: str) -> str:
        """读取Markdown文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_html(self, file_path: str) -> str:
        """读取HTML文件"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text()
        except ImportError:
            logger.warning("beautifulsoup4 未安装，回退到 SimpleDirectoryReader")
            return self._read_with_llama_index(file_path)

    def _read_with_llama_index(self, file_path: str) -> str:
        """使用LlamaIndex通用解析器读取文件"""
        from llama_index.core import SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        return "\n\n".join([doc.text for doc in documents])

    def _chunk_text(self, content: str, file_path: str = "") -> List[Dict[str, Any]]:
        """对文本进行分块

        Args:
            content: 文本内容
            file_path: 文件路径（用于元数据）

        Returns:
            分块结果列表

        Raises:
            ChunkingError: 分块失败
        """
        # 检查缓存
        content_hash = ChunkCache.generate_cache_key(content)
        cached_chunks = self._chunk_cache.get(content_hash)
        if cached_chunks is not None:
            logger.info(f"使用缓存的分块结果: {len(cached_chunks)} 个块")
            return cached_chunks

        try:
            # 执行分块
            chunk_objects = self._chunker.chunk_text(content, file_path)

            # 转换为字典格式
            chunks = []
            for chunk_obj in chunk_objects:
                chunks.append({
                    "text": chunk_obj.text,
                    "chunk_id": chunk_obj.chunk_id,
                    "metadata": chunk_obj.metadata,
                    "title": chunk_obj.title,
                    "position": chunk_obj.position
                })

            # 缓存结果
            self._chunk_cache.put(content_hash, chunks)
            logger.info(f"分块完成: {len(chunks)} 个块")

            return chunks

        except Exception as e:
            logger.error(f"分块失败: {e}")
            raise ChunkingError(f"分块失败: {str(e)}") from e

    def process_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """批量处理文件

        Args:
            file_paths: 文件路径列表

        Returns:
            处理结果列表
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except DocumentParseError as e:
                logger.error(f"处理文件失败: {file_path}, {e}")
                results.append({
                    "status": "error",
                    "file_path": file_path,
                    "error": str(e)
                })
        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self._chunk_cache.get_stats()

    def clear_cache(self) -> None:
        """清空缓存"""
        self._chunk_cache.clear()
        logger.info("文档处理缓存已清空")


def create_document_processor(
    chunker: Optional[IntelligentChunker] = None,
    chunk_cache: Optional[ChunkCache] = None
) -> DocumentProcessor:
    """创建文档处理器的工厂函数

    Args:
        chunker: 分块器实例
        chunk_cache: 分块缓存实例

    Returns:
        DocumentProcessor 实例
    """
    return DocumentProcessor(chunker=chunker, chunk_cache=chunk_cache)


# 导出
__all__ = [
    'DocumentProcessor',
    'create_document_processor',
]