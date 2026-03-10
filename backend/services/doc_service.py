"""文档管理服务模块 - 优化版，修复删除同步问题"""

import os
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag.core.engine import RAGEngine
from backend.config import config
from backend.logging_config import get_logger
from backend.exceptions import (
    UnsupportedFileTypeError,
    DocumentNotFoundError,
    DocumentParseError,
)

logger = get_logger(__name__)


class RebuildTask:
    """重建索引任务状态"""
    def __init__(self):
        self.status: str = "idle"  # idle, running, completed, failed
        self.progress: int = 0
        self.total: int = 0
        self.message: str = ""
        self.error: str = ""
        self._lock = threading.Lock()


# 全局重建任务状态
_rebuild_task = RebuildTask()

# 全局文件哈希缓存（模块级别，跨请求保持）
_file_hash_cache: Dict[str, str] = {}


class DocService:
    """文档管理服务类 - 优化版"""

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.docs_dir = config.get_docs_dir()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的SHA256哈希值（带缓存）
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件的SHA256哈希值
        """
        global _file_hash_cache
        
        # 检查缓存（使用绝对路径作为键）
        abs_path = str(Path(file_path).resolve())
        if abs_path in _file_hash_cache:
            return _file_hash_cache[abs_path]
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # 分块读取，避免大文件内存问题
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        file_hash = sha256_hash.hexdigest()
        _file_hash_cache[abs_path] = file_hash
        return file_hash
    
    def _get_existing_hashes(self) -> set:
        """获取知识库中所有现有文件的哈希值集合（使用缓存优化）"""
        global _file_hash_cache
        
        hashes = set()
        
        if not self.docs_dir.exists():
            return hashes
        
        for file_path in self.docs_dir.iterdir():
            if file_path.is_file():
                try:
                    abs_path = str(file_path.resolve())
                    # 优先使用缓存
                    if abs_path in _file_hash_cache:
                        hashes.add(_file_hash_cache[abs_path])
                    else:
                        # 不在缓存中才计算
                        file_hash = self._calculate_file_hash(str(file_path))
                        hashes.add(file_hash)
                except Exception as e:
                    logger.warning(f"计算文件哈希失败 {file_path}: {e}")
        
        return hashes
    
    def _get_file_size(self, file_path: str) -> int:
        """获取文件大小（字节）
        
        Args:
            file_path: 文件路径
            
        Returns:
            int: 文件大小
        """
        return os.path.getsize(file_path)

    def upload_document(self, file, file_name: str) -> Dict[str, Any]:
        """上传文档

        Args:
            file: 文件路径（str）或 Streamlit 上传的文件对象
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

            # 2. 临时保存文件用于大小检查和哈希计算
            import tempfile
            import shutil
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                if isinstance(file, str):
                    shutil.copy(file, tmp.name)
                    file_size = self._get_file_size(file)
                else:
                    content = file.getvalue()
                    tmp.write(content)
                    file_size = len(content)
                temp_file_path = tmp.name
            
            try:
                # 2.1 检查文件大小
                max_size_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
                if file_size > max_size_bytes:
                    max_size_mb = config.MAX_FILE_SIZE_MB
                    actual_size_mb = file_size / (1024 * 1024)
                    return {
                        "status": "error",
                        "message": f"文件大小超限：{actual_size_mb:.1f}MB > {max_size_mb}MB。请上传不超过 {max_size_mb}MB 的文件。"
                    }
                
                # 2.2 检查重复文件（基于内容哈希）
                if config.ENABLE_DUPLICATE_CHECK:
                    file_hash = self._calculate_file_hash(temp_file_path)
                    existing_hashes = self._get_existing_hashes()
                    
                    logger.info(f"DEBUG: 上传文件哈希: {file_hash}")
                    logger.info(f"DEBUG: 已有文件哈希集合: {existing_hashes}")
                    logger.info(f"DEBUG: 配置 ENABLE_DUPLICATE_CHECK = {config.ENABLE_DUPLICATE_CHECK}")
                    
                    if file_hash in existing_hashes:
                        logger.warning(f"检测到重复文件: {file_name} (哈希: {file_hash[:16]}...)")
                        return {
                            "status": "error",
                            "message": f"文件已存在：检测到内容相同的文件，请勿重复上传。"
                        }
                
                # 3. 保存文件到文档目录
                file_path = self.docs_dir / file_name

                # 处理文件名冲突
                counter = 1
                original_name = file_path.stem
                while file_path.exists():
                    file_path = self.docs_dir / f"{original_name}_{counter}{file_ext}"
                    counter += 1

                # 从临时文件复制到目标位置
                shutil.copy(temp_file_path, file_path)

            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

            # 4. 解析并向量化
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
                    "size": stat.st_size,  # 原始字节大小（整数）
                    "size_formatted": self._format_file_size(stat.st_size),  # 格式化大小（字符串）
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
            
            # 清理 hash 缓存
            abs_path = str(Path(file_path).resolve())
            global _file_hash_cache
            if abs_path in _file_hash_cache:
                del _file_hash_cache[abs_path]

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

    def rebuild_index(self, async_mode: bool = True) -> Dict[str, Any]:
        """重建索引

        Args:
            async_mode: 是否异步执行（默认True）

        Returns:
            Dict: 重建结果
        """
        global _rebuild_task
        
        # 如果已经在运行，返回当前状态
        if _rebuild_task.status == "running":
            return {
                "status": "running",
                "message": f"重建索引正在进行中... ({_rebuild_task.progress}/{_rebuild_task.total})",
                "progress": _rebuild_task.progress,
                "total": _rebuild_task.total
            }
        
        if async_mode:
            # 异步执行：启动后台线程
            _rebuild_task.status = "running"
            _rebuild_task.progress = 0
            _rebuild_task.message = "正在启动重建任务..."
            _rebuild_task.error = ""
            
            thread = threading.Thread(target=self._rebuild_index_worker, daemon=True)
            thread.start()
            
            return {
                "status": "started",
                "message": "重建任务已启动，正在后台处理..."
            }
        else:
            # 同步执行（保留兼容性）
            return self._rebuild_index_worker()

    def _rebuild_index_worker(self) -> Dict[str, Any]:
        """重建索引的实际工作线程"""
        global _rebuild_task
        
        try:
            with _rebuild_task._lock:
                _rebuild_task.status = "running"
                _rebuild_task.message = "正在清空索引..."
                _rebuild_task.progress = 0
            
            # 清空现有索引
            self.rag_engine.clear_index()

            # 重新添加所有文档
            documents = []
            for file_path in self.docs_dir.iterdir():
                if file_path.is_file():
                    documents.append(str(file_path))

            if not documents:
                with _rebuild_task._lock:
                    _rebuild_task.status = "completed"
                    _rebuild_task.message = "索引重建成功（无文档）"
                return {
                    "status": "success",
                    "message": "索引重建成功（无文档）"
                }

            with _rebuild_task._lock:
                _rebuild_task.total = len(documents)
                _rebuild_task.message = f"正在处理文档..."

            # 逐个添加文档
            total_chunks = 0
            for i, doc_path in enumerate(documents):
                with _rebuild_task._lock:
                    _rebuild_task.progress = i + 1
                    _rebuild_task.message = f"正在处理文档 {i+1}/{len(documents)}: {os.path.basename(doc_path)}"
                
                try:
                    result = self.rag_engine.add_documents(doc_path)
                    total_chunks += result.get("doc_count", 0)
                except Exception as doc_err:
                    logger.warning(f"处理文档失败 {doc_path}: {doc_err}")
                    # 继续处理其他文档
            
            with _rebuild_task._lock:
                _rebuild_task.status = "completed"
                _rebuild_task.message = f"索引重建成功，共处理 {len(documents)} 个文档，{total_chunks} 个文档块"

            return {
                "status": "success",
                "message": f"索引重建成功，共处理 {len(documents)} 个文档，{total_chunks} 个文档块"
            }

        except Exception as e:
            logger.error(f"重建索引失败: {e}", exc_info=True)
            with _rebuild_task._lock:
                _rebuild_task.status = "failed"
                _rebuild_task.error = str(e)
            return {
                "status": "error",
                "message": f"重建失败：{str(e)}"
            }

    def get_rebuild_status(self) -> Dict[str, Any]:
        """获取重建任务状态"""
        with _rebuild_task._lock:
            return {
                "status": _rebuild_task.status,
                "progress": _rebuild_task.progress,
                "total": _rebuild_task.total,
                "message": _rebuild_task.message,
                "error": _rebuild_task.error
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
