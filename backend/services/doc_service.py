"""文档管理服务模块 - 优化版，修复删除同步问题"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag.engine import RAGEngine
from backend.config import config
from backend.logging_config import get_logger
from backend.exceptions import (
    UnsupportedFileTypeError,
    DocumentNotFoundError,
    DocumentParseError,
)

logger = get_logger(__name__)


class DocService:
    """文档管理服务类 - 优化版"""

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.docs_dir = config.get_docs_dir()

    def upload_document(self, file, file_name: str) -> Dict[str, Any]:
        """上传文档

        Args:
            file: Streamlit上传的文件对象
            file_name: 文件名

        Returns:
            Dict: 上传结果
        """
        try:
            # 1. 检查文件类型
            allowed_extensions = {'.pdf', '.docx', '.txt', '.html', '.htm', '.md'}
            file_ext = Path(file_name).suffix.lower()

            if file_ext not in allowed_extensions:
                return {
                    "status": "error",
                    "message": f"不支持的文件类型：{file_ext}。支持的类型：{', '.join(allowed_extensions)}"
                }

            # 2. 保存文件
            file_path = self.docs_dir / file_name

            # 处理文件名冲突
            counter = 1
            original_name = file_path.stem
            while file_path.exists():
                file_path = self.docs_dir / f"{original_name}_{counter}{file_ext}"
                counter += 1

            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            # 3. 解析并向量化
            result = self.rag_engine.add_documents(str(file_path))

            return {
                "status": "success",
                "file_name": file_path.name,
                "message": f"文档上传成功，已加入知识库（{result['doc_count']}个文档块）"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"上传失败：{str(e)}"
            }

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出已上传的文档

        Returns:
            List: 文档列表
        """
        documents = []

        if not self.docs_dir.exists():
            return documents

        for file_path in self.docs_dir.iterdir():
            if file_path.is_file():
                # 获取文件信息
                stat = file_path.stat()
                documents.append({
                    "name": file_path.name,
                    "size": self._format_file_size(stat.st_size),
                    "modified": self._format_date(stat.st_mtime),
                    "type": file_path.suffix.lower()
                })

        return sorted(documents, key=lambda x: x["modified"], reverse=True)

    def delete_document(self, file_name: str) -> Dict[str, Any]:
        """删除文档 - 同步删除向量数据

        Args:
            file_name: 文件名

        Returns:
            Dict: 删除结果
        """
        try:
            file_path = self.docs_dir / file_name

            if not file_path.exists():
                return {
                    "status": "error",
                    "message": "文件不存在"
                }

            # 获取文件路径用于删除对应的向量
            file_path_str = str(file_path)

            # 删除文件
            file_path.unlink()

            # 同步删除ChromaDB中的向量
            self._delete_vectors_by_file(file_path_str)

            logger.info(f"文档 {file_name} 及其向量已删除")

            return {
                "status": "success",
                "message": f"文档 {file_name} 删除成功（向量已同步清理）"
            }

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return {
                "status": "error",
                "message": f"删除失败：{str(e)}"
            }

    def _delete_vectors_by_file(self, file_path: str) -> None:
        """根据文件路径删除对应的向量数据"""
        try:
            collection = self.rag_engine._get_collection()

            # 获取所有数据
            try:
                all_data = collection.get()
                if not all_data or not all_data.get('ids'):
                    return

                # 找出匹配的文件路径
                ids_to_delete = []
                for i, metadata in enumerate(all_data.get('metadatas', [])):
                    if metadata and metadata.get('file_path') == file_path:
                        ids_to_delete.append(all_data['ids'][i])

                # 批量删除
                if ids_to_delete:
                    collection.delete(ids=ids_to_delete)
                    logger.info(f"删除了 {len(ids_to_delete)} 个向量")
            except Exception as e:
                logger.warning(f"查询向量数据失败: {e}")

        except Exception as e:
            logger.warning(f"删除向量失败: {e}")
            # 不抛出异常，因为文件已删除

    def rebuild_index(self) -> Dict[str, Any]:
        """重建索引

        Returns:
            Dict: 重建结果
        """
        try:
            # 清空现有索引
            self.rag_engine.clear_index()

            # 重新添加所有文档
            documents = []
            for file_path in self.docs_dir.iterdir():
                if file_path.is_file():
                    documents.append(str(file_path))

            if not documents:
                return {
                    "status": "success",
                    "message": "索引重建成功（无文档）"
                }

            # 逐个添加文档
            total_chunks = 0
            for doc_path in documents:
                result = self.rag_engine.add_documents(doc_path)
                total_chunks += result.get("doc_count", 0)

            return {
                "status": "success",
                "message": f"索引重建成功，共处理 {len(documents)} 个文档，{total_chunks} 个文档块"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"重建失败：{str(e)}"
            }

    def clear_knowledge_base(self) -> Dict[str, Any]:
        """清空知识库

        Returns:
            Dict: 清空结果
        """
        try:
            # 清空向量索引
            self.rag_engine.clear_index()

            # 清空文档目录
            if self.docs_dir.exists():
                for file_path in self.docs_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()

            return {
                "status": "success",
                "message": "知识库已清空"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"清空失败：{str(e)}"
            }

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息

        Returns:
            Dict: 统计信息
        """
        docs = self.list_documents()
        total_size = sum(
            f.stat().st_size
            for f in self.docs_dir.iterdir()
            if f.is_file()
        ) if self.docs_dir.exists() else 0

        return {
            "document_count": len(docs),
            "total_size": self._format_file_size(total_size),
            "indexed_chunks": self.rag_engine.get_document_count()
        }

    @staticmethod
    def _format_file_size(size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @staticmethod
    def _format_date(timestamp: float) -> str:
        """格式化日期"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
