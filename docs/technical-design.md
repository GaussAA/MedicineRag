# 医疗知识问答系统 - 技术方案设计

## 一、系统架构设计

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户界面层 (Streamlit)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  问答输入框  │  │  回答展示区  │  │    参考来源/免责声明     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         业务逻辑层 (Backend)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  问答服务    │  │  文档管理服务 │  │     安全服务          │   │
│  │  QAService   │  │  DocService  │  │   SecurityService    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  敏感词过滤  │  │  日志服务     │  │     配置管理          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RAG引擎层 (LlamaIndex)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  文档加载器  │  │  文本分块器   │  │     向量检索器        │   │
│  │  Loaders    │  │  TextSplitter│  │    VectorRetriever   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Chroma        │  │   Ollama        │  │   本地文件系统   │
│   向量数据库    │  │   LLM服务       │  │   (文档存储)     │
│                 │  │                 │  │                 │
│  - 医学知识向量 │  │  - qwen3:8b    │  │  - PDF/Word/TXT │
│  - 元数据索引  │  │  - BGE-M3      │  │  - 原始文档     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 1.2 分层说明

| 层级       | 职责                           | 技术选型        |
| ---------- | ------------------------------ | --------------- |
| 用户界面层 | Web交互、输入输出展示          | Streamlit       |
| 业务逻辑层 | 核心业务处理、流程控制         | Python          |
| RAG引擎层  | 文档处理、向量检索、Prompt构建 | LlamaIndex      |
| 基础设施层 | 数据存储、模型推理             | Chroma + Ollama |

---

## 二、技术选型详解

### 2.1 前端：Streamlit

**为什么选择Streamlit**：

- 纯Python开发，无需前端知识
- 内置文件上传、文本输入、表格展示组件
- 支持Markdown渲染，完美适配问答场景
- 适合快速原型开发，毕设时间紧迫

**替代方案**（如需更好的UI）：

- Gradio：更轻量，但定制性稍弱
- React + FastAPI：更专业，但开发周期长

### 2.2 RAG编排：LlamaIndex

**为什么选择LlamaIndex**：

- 专为RAG场景设计，API简洁
- 内置多种文档加载器（PDF、Word、HTML等）
- 支持多种向量数据库
- 与Ollama集成良好

**核心组件**：

```python
# 文档加载
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("data/").load_data()

# 文本分块（优化方案：语义切片）
# MVP阶段使用固定长度分块
from llama_index.core import SentenceSplitter
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# 进阶方案：语义切片（避免疾病"病因"和"治疗"被切断）
# from llama_index.core import SemanticSplitterNodeParser
# embed_model = OllamaEmbedding(model="bge-m3")
# splitter = SemanticSplitterNodeParser(
#     embed_model=embed_model,
#     buffer_size=1,
#     breakpoint_percentile_threshold=95
# )

# 向量存储
from llama_index.vector_stores.chroma import ChromaVectorStore
db = ChromaVectorStore(persist_directory="./chroma_db")

# 检索引擎
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents, vector_store=db)
```

### 2.3 向量数据库：Chroma

**为什么选择Chroma**：

- 本地部署，无需额外服务
- 安装简单（uv pip install chromadb）
- 性能足够满足小规模知识库
- 与LlamaIndex深度集成

**数据模型**：

```python
# Chroma集合结构
{
    "ids": ["doc_1", "doc_2", ...],
    "embeddings": [[...], [...], ...],  # BGE-M3向量
    "documents": ["文本内容1", "文本内容2", ...],
    "metadatas": [
        {"source": "medical_guide.pdf", "page": 1, "disease": "糖尿病"},
        {"source": "drug_guide.pdf", "page": 5, "drug": "阿莫西林"},
        ...
    ]
}
```

### 2.4 LLM：Ollama qwen3:8b

**为什么选择qwen3:8b**：

- 中文理解能力强
- 8B参数，RTX 3050（8GB显存）可运行
- 本地部署，无API费用
- 响应速度快

**硬件优化配置**：

```bash
# 限制模型使用GPU显存，避免OOM
# 在Ollama启动参数中设置
OLLAMA_NUM_GPU=1  # 使用GPU数量
OLLAMA_MAX_LOADED_MODELS=1  # 最多同时加载1个模型
```

> **注意**：RTX 3050 8GB运行qwen3:8b + BGE-M3约需6GB显存，建议逐个加载模型（问答时加载LLM，文档向量化时加载Embedding模型），避免同时运行导致OOM。

**Ollama安装与模型下载**：

```bash
# 安装Ollama（Windows通过官方客户端）
# 下载模型
ollama pull qwen3:8b
ollama pull bge-m3
```

### 2.5 Embedding：BGE-M3

**为什么选择BGE-M3**：

- 中文Embedding效果最好的模型之一
- 支持多语言
- 兼顾稠密向量和稀疏向量
- Ollama本地运行

---

## 三、核心模块设计

### 3.1 模块架构

```
medical_qa_system/
├── app/                      # Streamlit应用
│   ├── __init__.py
│   ├── main.py               # 主入口
│   ├── pages/                # 页面组件
│   │   ├── __init__.py
│   │   ├── chat.py           # 问答页面
│   │   └── knowledge.py      # 知识库管理页面
│   └── components/           # 可复用组件
│       ├── __init__.py
│       ├── chat_input.py     # 聊天输入框
│       └── message.py        # 消息展示
│
├── backend/                  # 后端服务
│   ├── __init__.py
│   ├── config.py             # 配置管理
│   ├── services/             # 业务服务
│   │   ├── __init__.py
│   │   ├── qa_service.py     # 问答服务
│   │   ├── doc_service.py   # 文档管理服务
│   │   └── security_service.py  # 安全服务
│   └── utils/                # 工具函数
│       ├── __init__.py
│       ├── logger.py         # 日志工具
│       └──敏感词_filter.py   # 敏感词过滤
│
├── rag/                     # RAG引擎
│   ├── __init__.py
│   ├── engine.py             # RAG引擎核心
│   ├── prompts.py            # Prompt模板
│   └── settings.py           # RAG配置
│
├── data/                     # 数据目录
│   ├── documents/            # 原始文档
│   ├── chroma_db/           # Chroma向量库
│   └── logs/                # 日志文件
│
├── requirements.txt          # 依赖
└── .env                      # 环境变量
```

### 3.2 核心服务设计

#### 3.2.1 问答服务（QAService）

**职责**：处理用户问答请求

```python
# backend/services/qa_service.py

from typing import Optional
from pydantic import BaseModel

class QARequest(BaseModel):
    """问答请求"""
    question: str
    chat_history: Optional[list] = None  # 多轮对话历史

class QAResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: list[dict]  # 参考来源
    disclaimer: str      # 免责声明

class QAService:
    def __init__(self, rag_engine, security_service):
        self.rag_engine = rag_engine
        self.security_service = security_service
    
    def ask(self, request: QARequest) -> QAResponse:
        # 1. 敏感词检查
        check_result = self.security_service.check_content(request.question)
        if not check_result.is_safe:
            return QAResponse(
                answer=check_result.warning_message,
                sources=[],
                disclaimer=DISCLAIMER_TEXT
            )
        
        # 2. 构建Prompt
        prompt = self.rag_engine.build_prompt(
            query=request.question,
            history=request.chat_history
        )
        
        # 3. 检索相关文档
        retrieved_docs = self.rag_engine.retrieve(request.question, top_k=3)
        
        # 4. 调用LLM生成回答
        answer = self.rag_engine.generate(prompt, retrieved_docs)
        
        # 5. 提取来源信息（处理score可能为None或距离值的情况）
        sources = []
        for doc in retrieved_docs:
            # 处理score：可能是None、距离值（越小越相似）或相似度（越大越相似）
            score_value = doc.score
            if score_value is not None:
                # 归一化为百分比形式（假设是距离值，转为相似度）
                display_score = max(0, min(100, (1 - score_value) * 100))
            else:
                display_score = None
            
            sources.append({
                "content": doc.text[:200] + "...",
                "source": doc.metadata.get("source", "未知"),
                "score": display_score
            })
        
        # 6. 置信度检查：最高分低于阈值时，提示用户"未找到确切依据"
        if retrieved_docs and retrieved_docs[0].score and retrieved_docs[0].score > 0.5:
            # 距离值大于0.5表示相关性较低
            answer = "抱歉，在我的知识库中未找到足够准确的依据。为了您的健康，建议咨询专业医生。\n\n" + answer
        
        return QAResponse(
            answer=answer,
            sources=sources,
            disclaimer=DISCLAIMER_TEXT
        )
```

#### 3.2.2 文档管理服务（DocService）

**职责**：知识库构建与维护

```python
# backend/services/doc_service.py

import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from rag.engine import RAGEngine

class DocService:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.docs_dir = Path("data/documents")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
    
    def upload_document(self, file, file_name: str) -> dict:
        """上传文档"""
        # 1. 保存文件
        file_path = self.docs_dir / file_name
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        
        # 2. 解析并向量化
        self.rag_engine.add_documents(str(file_path))
        
        return {
            "status": "success",
            "file_name": file_name,
            "message": "文档上传成功，已加入知识库"
        }
    
    def list_documents(self) -> list[dict]:
        """列出知识库中的文档"""
        return self.rag_engine.list_indexed_documents()
    
    def delete_document(self, doc_id: str) -> dict:
        """删除文档"""
        self.rag_engine.delete_document(doc_id)
        return {"status": "success", "message": "文档删除成功"}
    
    def rebuild_index(self) -> dict:
        """重建索引"""
        self.rag_engine.rebuild_index()
        return {"status": "success", "message": "索引重建成功"}
```

#### 3.2.3 安全服务（SecurityService）

**职责**：敏感词过滤、内容安全检查

```python
# backend/services/security_service.py

import re
from typing import List

# 敏感词库（简化示例，实际使用更完整的词库）
SENSITIVE_WORDS = {
    "suicide": {"category": "自杀自残", "message": "如果您有自杀想法，请拨打心理援助热线400-161-9995"},
    "self_harm": {"category": "自杀自残", "message": "请保护好自己，必要时请拨打120或报警"},
    "violence": {"category": "暴力倾向", "message": "如遇紧急情况，请拨打110报警"},
    "drugs": {"category": "非法药物", "message": "请拨打戒毒热线400-658-6699"},
    "porn": {"category": "色情内容", "message": "无法回答此问题，请咨询其他话题"}
}

class CheckResult:
    def __init__(self, is_safe: bool, category: str = None, warning_message: str = None):
        self.is_safe = is_safe
        self.category = category
        self.warning_message = warning_message

class SecurityService:
    def __init__(self):
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> dict:
        """编译正则表达式"""
        return {
            category: re.compile(f"({keywords})", re.IGNORECASE)
            for category, keywords in self._get_keyword_patterns().items()
        }
    
    def _get_keyword_patterns(self) -> dict:
        return {
            "suicide": "自杀|自杀|割腕|服毒|跳楼|轻生",
            "self_harm": "自残|伤害自己",
            "violence": "杀人|伤害他人|打架|行凶",
            "drugs": "毒品|吸毒|贩毒|制毒",
            "porn": "色情|裸聊|性交易"
        }
    
    def check_content(self, text: str) -> CheckResult:
        """检查内容是否安全"""
        for category, pattern in self.patterns.items():
            if pattern.search(text):
                info = SENSITIVE_WORDS.get(category, {})
                return CheckResult(
                    is_safe=False,
                    category=info.get("category", category),
                    warning_message=info.get("message", "无法回答此问题")
                )
        
        return CheckResult(is_safe=True)
    
    def desensitize(self, text: str) -> str:
        """脱敏处理"""
        # 去除身份证号
        text = re.sub(r'\d{17}[\dXx]', '****', text)
        # 去除手机号
        text = re.sub(r'1[3-9]\d{9}', '***********', text)
        # 去除姓名（简单处理）
        text = re.sub(r'[张李王刘陈杨赵黄周吴郑孙朱马胡林郭何高罗]{1,2}先生|女士', '***', text)
        return text
```

---

## 四、数据模型设计

### 4.1 向量数据库集合

```python
# Chroma集合定义

collection_name = "medical_knowledge"

# 向量维度：BGE-M3输出为1024维
dimension = 1024

# 元数据结构
metadata_schema = {
    "source": "string",           # 文档来源文件名
    "doc_id": "string",          # 文档唯一ID
    "page": "int",               # 页码
    "chunk_id": "int",           # 分块ID
    "disease": "string",         # 关联疾病（可选）
    "drug": "string",            # 关联药物（可选）
    "category": "string",        # 知识类别：症状/疾病/用药/检查
    "created_at": "datetime"    # 入库时间
}
```

### 4.2 本地存储结构

```
data/
├── documents/                   # 原始文档
│   ├── medical_guide.pdf
│   ├── drug_manual.txt
│   └── ...
│
├── chroma_db/                   # Chroma向量库
│   ├── chroma.sqlite3
│   └── ...
│
├── conversation_history.json    # 对话历史（可选）
│
└── config.json                  # 配置文件
```

### 4.3 对话历史结构

```json
// conversation_history.json
{
    "sessions": [
        {
            "session_id": "uuid-1",
            "created_at": "2026-03-07T10:00:00",
            "messages": [
                {
                    "role": "user",
                    "content": "头痛是什么原因？",
                    "timestamp": "2026-03-07T10:00:01"
                },
                {
                    "role": "assistant", 
                    "content": "头痛可能由多种原因引起...",
                    "sources": [{"source": "medical_guide.pdf", "page": 5}],
                    "timestamp": "2026-03-07T10:00:05"
                }
            ]
        }
    ]
}
```

---

## 五、API接口设计

### 5.1 Streamlit页面路由

| 路径       | 方法 | 功能          | 页面         |
| ---------- | ---- | ------------- | ------------ |
| /          | GET  | 首页/问答主页 | chat.py      |
| /knowledge | GET  | 知识库管理    | knowledge.py |

### 5.2 内部服务接口

```python
# 问答接口
class QAEndpoints:
    @staticmethod
    def ask(question: str, history: list = None) -> dict:
        """
        POST /api/qa
        Body: {"question": "...", "history": [...]}
        Response: {"answer": "...", "sources": [...], "disclaimer": "..."}
        """

# 文档管理接口
class DocEndpoints:
    @staticmethod
    def upload(file) -> dict:
        """
        POST /api/docs/upload
        Body: multipart/form-data
        Response: {"status": "success", "file_name": "..."}
        """
    
    @staticmethod
    def list() -> list:
        """
        GET /api/docs/list
        Response: [{"doc_id": "...", "source": "...", "chunks": 10}]
        """
    
    @staticmethod
    def delete(doc_id: str) -> dict:
        """
        DELETE /api/docs/{doc_id}
        Response: {"status": "success"}
        """
    
    @staticmethod
    def rebuild() -> dict:
        """
        POST /api/docs/rebuild
        Response: {"status": "success", "message": "索引重建成功"}
        """
```

### 5.3 Streamlit状态管理

```python
# 使用st.session_state管理状态
if "messages" not in st.session_state:
    st.session_state.messages = []  # 对话历史

if "history" not in st.session_state:
    st.session_state.history = []    # 多轮上下文

if "documents" not in st.session_state:
    st.session_state.documents = []  # 已上传文档列表
```

---

## 六、Prompt工程设计

### 6.1 基础Prompt模板

```python
# rag/prompts.py

SYSTEM_PROMPT = """您是一位专业的医疗健康助手。您的职责是根据提供的参考资料，用通俗易懂的语言回答用户的问题。

重要规则：
1. 只根据提供的参考文档回答，不要编造信息
2. 如果参考文档中没有相关信息，请明确告知用户
3. 回答要简洁清晰，使用Markdown格式
4. 每次回答必须附带免责声明
5. 如果问题涉及紧急症状，引导用户立即就医

回答格式要求：
- 先给出核心答案
- 然后提供详细解释
- 最后给出建议（如有）

免责声明：本回答仅供参考，不能替代医生诊断。如有严重症状，请立即就医。
紧急情况请拨打120急救电话。
"""

USER_PROMPT_TEMPLATE = """请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{question}

请给出回答："""
```

### 6.2 检索增强Prompt

```python
# 针对不同问题类型的Prompt优化

QUESTION_TYPE_PROMPTS = {
    "symptom": """用户询问症状相关问题。请从参考资料中提取：
1. 可能的原因
2. 伴随症状
3. 严重程度判断
4. 就医建议""",
    
    "disease": """用户询问疾病相关知识。请从参考资料中提取：
1. 疾病定义
2. 症状表现
3. 治疗方法
4. 预防措施""",
    
    "medication": """用户询问用药指导。请从参考资料中提取：
1. 适应症
2. 用法用量
3. 禁忌症
4. 副作用
5. 注意事项""",
    
    "examination": """用户询问检查报告解读。请从参考资料中提取：
1. 指标含义
2. 正常参考值
3. 异常可能原因
4. 进一步建议"""
}
```

---

## 七、关键代码实现

### 7.1 RAG引擎核心

```python
# rag/engine.py

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
import chromadb

class RAGEngine:
    def __init__(self, config: dict):
        # 初始化Embedding模型
        self.embed_model = OllamaEmbedding(
            model="bge-m3",
            base_url="http://localhost:11434"
        )
        
        # 初始化LLM
        self.llm = Ollama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.3,
            max_tokens=1024
        )
        
        # 配置LlamaIndex全局设置
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = config.get("chunk_size", 512)
        Settings.chunk_overlap = config.get("chunk_overlap", 50)
        
        # 初始化向量数据库
        self.db = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.db.get_or_create_collection("medical_knowledge")
        self.vector_store = ChromaVectorStore(
            chroma_client=self.db,
            collection_name="medical_knowledge"
        )
        
        # 加载或创建索引
        self.index = self._load_or_create_index()
    
    def _load_or_create_index(self):
        """加载或创建索引"""
        if self.collection.count() > 0:
            # 从向量库加载已有索引
            return VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
        else:
            # 创建新索引（首次运行时）
            return None
    
    def add_documents(self, file_path: str):
        """添加文档到知识库"""
        # 加载文档
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        
        # 创建索引并存储
        self.index = VectorStoreIndex.from_documents(
            documents,
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
    
    def retrieve(self, query: str, top_k: int = 3) -> list:
        """检索相关文档"""
        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            similarity_threshold=0.5
        )
        return retriever.retrieve(query)
    
    def generate(self, prompt: str, retrieved_docs: list) -> str:
        """生成回答"""
        # 构建上下文
        context = "\n\n".join([doc.text for doc in retrieved_docs])
        
        # 构造完整prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(context=context, question=prompt)}"
        
        # 调用LLM
        response = self.llm.complete(full_prompt)
        return response.text
    
    def build_prompt(self, query: str, history: list = None) -> str:
        """构建Prompt（含历史上下文）"""
        if history:
            context = "\n".join([
                f"用户：{h['question']}\n助手：{h['answer']}"
                for h in history[-3:]  # 保留最近3轮
            ])
            return f"对话历史：\n{context}\n\n当前问题：{query}"
        return query
```

### 7.2 Streamlit主界面

```python
# app/main.py

import streamlit as st
from backend.services.qa_service import QAService
from backend.services.doc_service import DocService
from backend.services.security_service import SecurityService
from rag.engine import RAGEngine

# 页面配置
st.set_page_config(
    page_title="医疗知识问答系统",
    page_icon="🏥",
    layout="wide"
)

# 初始化服务（单例）
@st.cache_resource
def init_services():
    config = {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 3,
        "similarity_threshold": 0.5
    }
    rag_engine = RAGEngine(config)
    security_service = SecurityService()
    qa_service = QAService(rag_engine, security_service)
    doc_service = DocService(rag_engine)
    return qa_service, doc_service

qa_service, doc_service = init_services()

# 页面标题
st.title("🏥 医疗知识问答系统")
st.markdown("### 基于RAG技术的医疗健康咨询")

# 侧边栏
with st.sidebar:
    st.header("📚 知识库管理")
    
    uploaded_file = st.file_uploader(
        "上传医疗文档",
        type=["pdf", "docx", "txt", "html"]
    )
    
    if uploaded_file and st.button("上传并构建索引"):
        with st.spinner("正在处理文档..."):
            result = doc_service.upload_document(uploaded_file, uploaded_file.name)
            st.success(result["message"])
    
    st.divider()
    
    if st.button("重建知识库索引"):
        with st.spinner("正在重建..."):
            result = doc_service.rebuild_index()
            st.success(result["message"])

# 主聊天区域
st.subheader("💬 问答咨询")

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 显示参考来源
        if "sources" in message and message["sources"]:
            with st.expander("📄 参考来源"):
                for src in message["sources"]:
                    st.markdown(f"- {src['source']}")

# 免责声明
st.warning("⚠️ 免责声明：本系统回答仅供参考，不能替代医生诊断。如有严重症状，请立即就医。紧急情况请拨打120。")

# 聊天输入
if prompt := st.chat_input("请输入您的医疗问题..."):
    # 用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            response = qa_service.ask(
                QARequest(question=prompt)
            )
            
            # 显示回答
            st.markdown(response.answer)
            
            # 显示参考来源
            if response.sources:
                with st.expander("📄 参考来源"):
                    for src in response.sources:
                        st.markdown(f"- **{src['source']}**：{src['content']}")
            
            # 强制显示免责声明
            st.error(response.disclaimer)
    
    # 保存到历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "sources": response.sources
    })
```

---

## 八、部署方案

### 8.1 本地部署

```bash
# 1. 克隆项目
git clone <project-url>
cd medical-qa-system

# 2. 创建虚拟环境并安装依赖（使用uv）
uv venv
source .venv/bin/activate
uv sync

# 3. 启动Ollama服务
# 方式1：后台运行Ollama
# 方式2：手动启动ollama serve

# 4. 下载模型
ollama pull qwen3:8b
ollama pull bge-m3

# 5. 启动Streamlit
streamlit run app/main.py
```

### 8.2 依赖清单

```toml
# pyproject.toml（推荐）

[project]
name = "medical-qa-system"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "streamlit>=1.30.0",
    "llama-index>=0.10.0",
    "llama-index-embeddings-ollama>=0.1.0",
    "llama-index-llms-ollama>=0.1.0",
    "llama-index-vector-stores-chroma>=0.1.0",
    "chromadb>=0.4.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    # 文档解析
    "pypdf>=3.0.0",
    "pymupdf>=1.23.0",  # 更好的PDF解析，支持表格提取
    "python-docx>=1.0.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=4.9.0",
    # "llamaparse>=0.1.0",  # 进阶：LlamaParse更擅长复杂PDF
]
```

> **PDF解析优化说明**：
> - MVP阶段：使用PyPDF（简单文本提取）
> - 进阶方案：使用PyMuPDF（支持表格提取、布局分析）
> - 高阶方案：使用LlamaParse（专为RAG设计，解析医疗说明书效果更好）
```

> **说明**：推荐使用 `uv` 管理项目，创建项目后执行 `uv sync` 即可快速安装依赖。

### 8.3 启动顺序

```
1. Ollama服务（后台）
   └─ 加载 qwen3:8b 模型
   └─ 加载 bge-m3 模型

2. Chroma向量数据库
   └─ 加载已有索引或创建新索引

3. Streamlit应用
   └─ 初始化RAG引擎
   └─ 启动Web服务（默认端口8501）
```

---

## 九、测试方案

### 9.1 功能测试

| 测试项     | 测试方法             | 预期结果                 |
| ---------- | -------------------- | ------------------------ |
| 文档上传   | 上传PDF文件          | 文件保存成功，向量库增加 |
| 问答功能   | 输入常见问题         | 返回相关回答             |
| 敏感词过滤 | 输入包含敏感词的问题 | 显示拦截提示             |
| 免责声明   | 任意问答后检查       | 显示免责声明             |
| 参考来源   | 检查回答来源         | 显示参考文档             |

### 9.2 性能测试

| 指标         | 测试方法         | 目标     |
| ------------ | ---------------- | -------- |
| 首次响应时间 | 冷启动后首次提问 | <30秒    |
| 热响应时间   | 模型加载后提问   | <15秒    |
| 检索相关性   | 抽查10个问题     | >80%相关 |

---

## 十、总结

本技术方案基于需求文档，针对本科毕设场景进行了优化：

1. **技术栈简洁**：Streamlit + LlamaIndex + Chroma + Ollama
2. **成本为零**：纯本地部署，无需API费用
3. **开发周期短**：1-2周可完成MVP
4. **可扩展性强**：后续可轻松升级到API版本

关键实现点：

- RAG引擎封装，便于替换底层组件
- 服务层分离，便于业务扩展
- 敏感词过滤保障医疗合规
- Prompt模板优化提升回答质量
- 置信度机制减少幻觉回答

---

## 十一、答辩加分项建议

### 1. 思维链展示（必做）
- 在界面添加"查看推理过程"开关
- 展示检索到的原文片段 + 模型推理过程
- 体现RAG技术原理，增加技术深度

### 2. 高级PDF解析
- 使用PyMuPDF处理表格数据
- 或使用LlamaParse解析复杂医疗说明书
- 体现对工程难点的理解

### 3. 多轮对话槽位填充
- 第一轮：用户说"头痛" → 系统反问"痛了多久？有没有发烧？"
- 收集完信息后再检索生成答案
- 体现"专家系统"思维

### 4. 混合检索实现
- 向量检索 + BM25关键词检索融合
- 解决医疗专有名词检索不准的问题
- 体现对检索算法的深入理解
