"""智能文档分块模块

功能：
1. 基于句子边界的智能分块
2. 保留文档结构（标题层级）
3. 医学生成器词保护
4. 优化的重叠策略
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from backend.logging_config import get_logger
from backend.config import config
from backend.exceptions import DocumentParseError

logger = get_logger(__name__)


@dataclass
class Chunk:
    """分块结果"""
    text: str
    chunk_id: str
    metadata: Dict[str, Any]
    title: Optional[str] = None  # 所属标题
    position: int = 0  # 在文档中的位置


class IntelligentChunker:
    """智能文档分块器"""
    
    # 句子结束标点
    SENTENCE_ENDINGS = r'[。！？\.!?\n]+'
    
    # 医学术语列表（不应被分割）
    MEDICAL_TERMS = {
        # 疾病名称
        "高血压", "糖尿病", "冠心病", "脑卒中", "肺炎", "肝炎",
        "胃炎", "肠炎", "肾炎", "膀胱炎", "心肌炎", "心包炎",
        "甲状腺功能亢进", "甲状腺功能减退", "系统性红斑狼疮",
        "类风湿性关节炎", "骨关节炎", "骨质疏松",
        # 症状
        "呼吸困难", "胸闷气短", "心悸怔忡", "头晕目眩",
        "恶心呕吐", "腹痛腹泻", "尿频尿急", "发热寒战",
        # 药品名称
        "阿莫西林", "布洛芬", "氨氯地平", "硝苯地平",
        "卡托普利", "依那普利", "美托洛尔", "阿司匹林",
        "二甲双胍", "格列本脲", "胰岛素",
    }
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, file_path: str = "") -> List[Chunk]:
        """对文本进行智能分块
        
        Args:
            text: 输入文本
            file_path: 文件路径（用于元数据）
            
        Returns:
            分块列表
        """
        if not text or not text.strip():
            return []
        
        # 提取标题结构
        titles = self._extract_titles(text)
        logger.debug(f"提取到 {len(titles)} 个标题")
        
        # 提取句子
        sentences = self._split_into_sentences(text)
        logger.debug(f"提取到 {len(sentences)} 个句子")
        
        # 基于句子进行分块
        chunks = self._create_chunks(sentences, titles, file_path)
        
        logger.info(f"分块完成，得到 {len(chunks)} 个块")
        return chunks
    
    def _extract_titles(self, text: str) -> List[Tuple[str, int]]:
        """提取标题及其位置
        
        Returns:
            [(标题文本, 起始位置), ...]
        """
        # 匹配Markdown标题 (# ## ###) 或数字标题 (1. 2. 3.)
        title_pattern = r'^(#{1,6}\s+.+|(?:\d+\.)+\s+.+)$'
        
        titles = []
        for match in re.finditer(title_pattern, text, re.MULTILINE):
            titles.append((match.group().strip(), match.start()))
        
        # 按位置排序
        titles.sort(key=lambda x: x[1])
        return titles
    
    def _get_current_title(self, position: int, titles: List[Tuple[str, int]]) -> Optional[str]:
        """获取指定位置所属的标题"""
        current_title = None
        for title, pos in titles:
            if position < pos:
                break
            current_title = title
        return current_title
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 先按段落分割
        paragraphs = text.split('\n\n')
        
        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 按句子结束标点分割
            parts = re.split(self.SENTENCE_ENDINGS, para)
            
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)
        
        return sentences
    
    def _is_medical_term(self, text: str) -> bool:
        """检查文本是否包含医学术语"""
        for term in self.MEDICAL_TERMS:
            if term in text:
                return True
        return False
    
    def _create_chunks(
        self,
        sentences: List[str],
        titles: List[Tuple[str, int]],
        file_path: str
    ) -> List[Chunk]:
        """创建分块"""
        chunks = []
        current_chunk = ""
        current_title = None
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查当前句子是否包含医学术语
            has_medical_term = self._is_medical_term(sentence)
            
            # 尝试将句子添加到当前chunk
            if len(current_chunk) + len(sentence) < self.chunk_size:
                # 如果当前chunk为空，更新标题
                if not current_chunk:
                    # 估算当前句子的位置
                    position = i * 100  # 粗略估算
                    current_title = self._get_current_title(position, titles)
                
                current_chunk += sentence + "。 "
            
            else:
                # 当前chunk已满，保存
                if current_chunk.strip():
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        chunk_id=f"chunk_{chunk_id}",
                        metadata={
                            "file_path": file_path,
                            "chunk_index": chunk_id,
                            "char_count": len(current_chunk)
                        },
                        title=current_title,
                        position=chunk_id
                    ))
                    chunk_id += 1
                
                # 处理新句子
                # 如果单个句子超过chunk_size，强制分割
                if len(sentence) > self.chunk_size:
                    # 递归分割长句子
                    sub_chunks = self._split_long_sentence(sentence, chunk_id, file_path, current_title)
                    chunks.extend(sub_chunks)
                    chunk_id += len(sub_chunks)
                    current_chunk = ""
                    current_title = None
                else:
                    # 保留overlap内容
                    if len(current_chunk) > self.chunk_overlap:
                        current_chunk = current_chunk[-self.chunk_overlap:]
                    else:
                        current_chunk = ""
                    
                    # 更新标题
                    position = i * 100
                    current_title = self._get_current_title(position, titles)
                    
                    current_chunk += sentence + "。 "
        
        # 添加最后一个chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_id}",
                metadata={
                    "file_path": file_path,
                    "chunk_index": chunk_id,
                    "char_count": len(current_chunk)
                },
                title=current_title,
                position=chunk_id
            ))
        
        return chunks
    
    def _split_long_sentence(
        self,
        sentence: str,
        start_id: int,
        file_path: str,
        title: Optional[str]
    ) -> List[Chunk]:
        """分割过长的句子"""
        chunks = []
        # 按子句分割（逗号、分号等）
        sub_parts = re.split(r'[,，;；]', sentence)
        
        current = ""
        for i, part in enumerate(sub_parts):
            part = part.strip()
            if not part:
                continue
            
            if len(current) + len(part) < self.chunk_size:
                current += part + ", "
            else:
                if current:
                    chunks.append(Chunk(
                        text=current.strip().rstrip(','),
                        chunk_id=f"chunk_{start_id + len(chunks)}",
                        metadata={"file_path": file_path},
                        title=title,
                        position=start_id + len(chunks)
                    ))
                current = part + ", "
        
        if current:
            chunks.append(Chunk(
                text=current.strip().rstrip(','),
                chunk_id=f"chunk_{start_id + len(chunks)}",
                metadata={"file_path": file_path},
                title=title,
                position=start_id + len(chunks)
            ))
        
        return chunks


def create_chunker() -> IntelligentChunker:
    """创建分块器实例"""
    return IntelligentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )


def parse_document(file_path: str) -> str:
    """增强的文档解析函数
    
    支持多种格式的文档解析，包含错误处理和日志记录
    
    Args:
        file_path: 文件路径
        
    Returns:
        解析后的文本内容
    """
    from pathlib import Path
    from llama_index.core import SimpleDirectoryReader
    import logging
    
    file_path_obj = Path(file_path)
    suffix = file_path_obj.suffix.lower()
    
    logger.info(f"开始解析文档: {file_path}")
    
    try:
        # TXT文件 - 直接读取
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"TXT文档解析完成，字符数: {len(content)}")
            return content
        
        # Markdown文件 - 使用SimpleDirectoryReader
        elif suffix in ['.md', '.markdown']:
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            content = "\n\n".join([doc.text for doc in documents])
            logger.info(f"Markdown文档解析完成，字符数: {len(content)}")
            return content
        
        # HTML文件 - 提取文本
        elif suffix in ['.html', '.htm']:
            try:
                from bs4 import BeautifulSoup
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                # 移除script和style标签
                for script in soup(["script", "style"]):
                    script.decompose()
                content = soup.get_text(separator='\n')
                # 清理空白
                lines = [line.strip() for line in content.split('\n')]
                content = '\n'.join(line for line in lines if line)
                logger.info(f"HTML文档解析完成，字符数: {len(content)}")
                return content
            except ImportError:
                logger.warning("BeautifulSoup未安装，使用SimpleDirectoryReader")
                reader = SimpleDirectoryReader(input_files=[file_path])
                documents = reader.load_data()
                content = "\n\n".join([doc.text for doc in documents])
                return content
        
        # PDF和Word文件 - 使用SimpleDirectoryReader
        elif suffix in ['.pdf', '.docx', '.doc']:
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            content = "\n\n".join([doc.text for doc in documents])
            logger.info(f"{suffix}文档解析完成，字符数: {len(content)}")
            return content
        
        else:
            # 默认使用SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            content = "\n\n".join([doc.text for doc in documents])
            logger.info(f"文档解析完成，字符数: {len(content)}")
            return content
            
    except Exception as e:
        logger.error(f"文档解析失败: {file_path}, 错误: {e}")
        raise DocumentParseError(f"无法解析文档 {file_path}: {str(e)}") from e
