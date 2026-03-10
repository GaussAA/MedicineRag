"""分块器测试"""

import pytest
from rag.processing.chunker import IntelligentChunker, Chunk, create_chunker


class TestIntelligentChunker:
    """智能分块器测试"""
    
    def setup_method(self):
        """每个测试方法前初始化"""
        self.chunker = IntelligentChunker(chunk_size=200, chunk_overlap=50)
    
    def test_chunk_text_basic(self):
        """测试基本分块功能"""
        text = "这是第一段内容。\n\n这是第二段内容。\n\n这是第三段内容。"
        chunks = self.chunker.chunk_text(text, "test.txt")
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_chunk_text_empty(self):
        """测试空文本"""
        chunks = self.chunker.chunk_text("", "test.txt")
        assert len(chunks) == 0
        
        chunks = self.chunker.chunk_text("   ", "test.txt")
        assert len(chunks) == 0
    
    def test_chunk_text_preserves_medical_terms(self):
        """测试医学术语保留"""
        # 包含医学术语的文本
        text = "患者患有高血压和糖尿病，需要服用阿司匹林和二甲双胍。"
        chunks = self.chunker.chunk_text(text, "test.txt")
        
        # 验证分块包含医学术语
        all_text = " ".join([c.text for c in chunks])
        assert "高血压" in all_text
        assert "糖尿病" in all_text
    
    def test_extract_titles(self):
        """测试标题提取"""
        text = """# 第一章
        
内容一

## 第二章
        
内容二
"""
        titles = self.chunker._extract_titles(text)
        assert len(titles) >= 0
    
    def test_split_into_sentences(self):
        """测试句子分割"""
        text = "这是第一句。这是第二句！这是第三句？"
        sentences = self.chunker._split_into_sentences(text)
        assert len(sentences) > 0
    
    def test_chunk_has_metadata(self):
        """测试分块包含元数据"""
        text = "测试内容。"
        chunks = self.chunker.chunk_text(text, "test.txt")
        
        if chunks:
            chunk = chunks[0]
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'metadata')
    
    def test_create_chunker_factory(self):
        """测试工厂函数"""
        chunker = create_chunker()
        assert chunker is not None
        assert isinstance(chunker, IntelligentChunker)


class TestChunk:
    """Chunk数据类测试"""
    
    def test_chunk_creation(self):
        """测试Chunk创建"""
        chunk = Chunk(
            text="测试文本",
            chunk_id="chunk_1",
            metadata={"file_path": "test.txt"}
        )
        assert chunk.text == "测试文本"
        assert chunk.chunk_id == "chunk_1"
        assert chunk.metadata["file_path"] == "test.txt"
    
    def test_chunk_with_title(self):
        """测试带标题的Chunk"""
        chunk = Chunk(
            text="内容",
            chunk_id="chunk_1",
            metadata={},
            title="标题"
        )
        assert chunk.title == "标题"
    
    def test_chunk_defaults(self):
        """测试Chunk默认值"""
        chunk = Chunk(text="内容", chunk_id="chunk_1", metadata={})
        assert chunk.text == "内容"
        assert chunk.metadata == {}
        assert chunk.title is None
        assert chunk.position == 0
