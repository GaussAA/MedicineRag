# 医疗知识问答系统 (MedicineRag)

## 项目概述

基于RAG（检索增强生成）技术的医疗知识问答系统，使用本地Ollama模型（Qwen3:8b + BGE-M3）进行问答，无需外接API，保护隐私。

### 技术栈

| 层级         | 技术选型                  |
| ------------ | ------------------------- |
| 前端框架     | Streamlit 1.30+           |
| RAG框架      | LlamaIndex 0.10+          |
| 向量数据库   | ChromaDB 0.4+             |
| LLM          | Ollama qwen3:8b           |
| Embedding    | BGE-M3:latest             |
| 重排序模型   | BGE-Reranker-v2-m3:latest |
| API框架      | FastAPI 0.100+            |
| python包管理 | uv                        |

---

## 项目结构

```
MedicineRag/
├── app/                    # Streamlit Web应用
│   ├── main.py             # 主问答页面
│   ├── data/               # Web应用数据目录
│   │   ├── chroma_db/      # Chroma向量数据库
│   │   └── documents/     # 上传的源文档
│   └── pages/
│       ├── knowledge.py    # 知识库管理页面
│       └── analytics.py    # 系统统计页面
├── backend/               # 后端服务
│   ├── config.py          # 配置管理（安全解析+限流配置）
│   ├── exceptions.py      # 自定义异常类
│   ├── logging_config.py  # 统一日志配置
│   ├── statistics.py      # 问答统计模块（异步批量写入）
│   ├── api/               # FastAPI服务
│   │   ├── main.py        # API入口（限流中间件+健康检查）
│   │   ├── models.py      # 数据模型
│   │   └── routes/        # API路由
│   │       ├── qa.py      # 问答API（依赖注入+流式输出）
│   │       └── docs.py    # 文档管理API
│   └── services/          # 服务层
│       ├── qa_service.py          # 问答服务
│       ├── doc_service.py         # 文档管理服务
│       └── security_service.py    # 安全检查服务
├── rag/                   # RAG核心引擎
│   ├── engine.py          # RAG引擎实现（单例模式+缓存优化）
│   ├── chunker.py         # 智能文档分块（增强解析）
│   ├── retriever.py       # 混合检索模块
│   ├── reranker.py        # 重排序模块（两阶段检索）
│   └── prompts.py         # Prompt模板
├── tests/                 # 单元测试
│   ├── test_config.py
│   ├── test_exceptions.py
│   ├── test_logging.py
│   ├── test_security.py
│   └── test_statistics.py
├── data/                  # 数据目录
│   ├── documents/         # 上传的源文档
│   ├── chroma_db/         # Chroma向量数据库
│   └── qa_stats.json      # 问答统计文件
├── .env                   # 环境配置
├── pyproject.toml         # 项目配置
├── test_rag.py           # RAG测试脚本
└── test_rerank.py        # 重排序测试脚本
```

---

## 快速开始

### 1. 环境准备

```bash
# 确保已安装Python 3.10+
python --version

# 确保已安装Ollama并启动
ollama serve

# 下载所需模型
ollama pull qwen3:8b
ollama pull bge-m3:latest
ollama pull dengcao/bge-reranker-v2-m3:latest
```

### 2. 安装依赖

```bash
# 使用uv安装
uv sync

# 或使用pip
pip install -e .
```

### 3. 运行应用

```bash
# 方式1：激活虚拟环境并运行Streamlit Web应用
source .venv/Scripts/activate
streamlit run app/main.py

# 方式2：运行API服务
uvicorn backend.api.main:app --reload --port 8000
```

访问 http://localhost:8501 即可使用Web应用。

---

## 核心功能

### 问答流程
1. 用户输入医疗问题
2. 安全检查（敏感词、紧急症状检测）
3. 向量检索（Chroma + BGE-M3）
4. **两阶段检索**：初始召回 → 重排序（BGE-Reranker-v2-m3）
5. LLM生成回答（Qwen3:8b）
6. 返回回答 + 参考来源 + 免责声明

### 知识库管理
- 支持PDF、Word、TXT、HTML、Markdown格式
- 智能分块、向量化存储
- 重建索引、清空知识库功能

### 高级功能

#### 两阶段检索（重排序）
- **第一阶段**：使用BGE-M3向量化检索，召回top_k=20个候选文档
- **第二阶段**：使用BGE-Reranker-v2-m3 Cross-Encoder对候选文档进行精排
- 返回top_k=5个最相关结果
- 可通过配置启用/禁用重排序功能

#### 流式输出
- LLM回答逐字流式显示，减少等待焦虑
- 支持非流式和流式两种模式

#### 多轮对话
- 支持上下文理解（5轮历史）
- 历史长度自动控制，避免上下文溢出

#### API限流保护
- 问答接口：30次/分钟
- 文档上传：10次/分钟
- 其他接口：60次/分钟
- 可通过环境变量配置

#### 可观测性
- 问答统计面板
- 问题类型分布分析
- 性能指标监控（/metrics端点）
- 知识库缺口识别

#### REST API
- 完整的RESTful API接口
- 支持文档上传、问答、统计等功能
- 自动生成API文档（Swagger UI）

---

## API接口

| 接口                 | 方法   | 功能         |
| -------------------- | ------ | ------------ |
| `/api/qa/ask`        | POST   | 问答接口     |
| `/api/qa/stream`     | POST   | 流式问答     |
| `/api/docs/upload`   | POST   | 上传文档     |
| `/api/docs/list`     | GET    | 文档列表     |
| `/api/docs/{doc_id}` | DELETE | 删除文档     |
| `/api/docs/rebuild`  | POST   | 重建索引     |
| `/api/docs/clear`    | POST   | 清空知识库   |
| `/health`            | GET    | 基础健康检查 |
| `/health/detailed`   | GET    | 详细健康检查 |
| `/metrics`           | GET    | 性能监控指标 |

访问 http://localhost:8000/docs 查看完整API文档。

---

## 配置说明

在 `.env` 文件中配置：

```bash
# Ollama配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=bge-m3:latest
OLLAMA_LLM_MODEL=qwen3:8b

# 重排序配置（新增）
OLLAMA_RERANK_MODEL=dengcao/bge-reranker-v2-m3:latest
ENABLE_RERANK=true
RERANK_INITIAL_TOP_K=20

# RAG配置
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
SIMILARITY_THRESHOLD=0.3

# LLM生成参数
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1536

# 缓存配置
ENABLE_EMBEDDING_CACHE=true
EMBEDDING_CACHE_SIZE=100

# 分块策略
CHUNK_BY_TITLE=true
PRESERVE_MEDICAL_TERMS=true

# API限流配置
ENABLE_RATE_LIMIT=true
RATE_LIMIT_QPS=10

# CORS配置（生产环境建议限制）
CORS_ALLOWED_ORIGINS=*
```

### 重排序配置说明

| 配置项                 | 默认值                            | 说明               |
| ---------------------- | --------------------------------- | ------------------ |
| `OLLAMA_RERANK_MODEL`  | dengcao/bge-reranker-v2-m3:latest | 重排序模型         |
| `ENABLE_RERANK`        | true                              | 是否启用重排序     |
| `RERANK_INITIAL_TOP_K` | 20                                | 初始召回的文档数量 |

---

## 关键实现

### RAG引擎 (rag/engine.py)

使用原生Ollama API避免HTTP代理连接池问题，采用单例模式，支持两阶段检索：

```python
from rag.reranker import create_reranker

class RAGEngine:
    _instance: Optional['RAGEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def retrieve(self, query: str, top_k: int = 5):
        # 第一阶段：初始召回
        collection = self._get_collection()
        retriever = create_retriever(self.ollama_client, collection)
        initial_top_k = self.reranker.initial_top_k if self.reranker.enabled else top_k
        docs = retriever.retrieve(query=query, top_k=initial_top_k)
        
        # 第二阶段：重排序
        if self.reranker.enabled and len(docs) > top_k:
            docs = self.reranker.rerank(query, docs, top_k)
        
        return docs
    
    def generate(self, query: str, retrieved_docs):
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        full_prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\n问题：{query}"
        response = self.ollama_client.chat(
            model=config.OLLAMA_LLM_MODEL,
            messages=[{'role': 'user', 'content': full_prompt}]
        )
        return response.message.content
```

### 重排序模块 (rag/reranker.py)

使用BGE-Reranker-v2-m3模型对检索结果进行重排序：

```python
class Reranker:
    def __init__(self, ollama_client):
        self.model = config.OLLAMA_RERANK_MODEL
        self.enabled = config.ENABLE_RERANK
        self.initial_top_k = config.RERANK_INITIAL_TOP_K
    
    def rerank(self, query: str, documents: List[Dict], top_k: int):
        # 获取查询和文档的embedding
        query_embedding = self._get_embedding(query)
        
        # 计算每个文档的相关性分数
        scored_docs = []
        for doc in documents:
            doc_embedding = self._get_embedding(doc['text'])
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            new_doc = doc.copy()
            new_doc['rerank_score'] = similarity
            scored_docs.append(new_doc)
        
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        return scored_docs[:top_k]
```

### 统计模块 (backend/statistics.py)

异步批量写入，避免频繁I/O：

```python
class QAStats:
    def __init__(self, stats_file: str = "data/qa_stats.json"):
        self._lock = threading.Lock()
        self._dirty = False
        self._flush_interval = 30  # 30秒刷新一次
        self._max_records_before_flush = 100
```

### API限流 (backend/api/main.py)

基于客户端IP的内存限流器：

```python
class RateLimiter:
    def is_allowed(self, client_id: str, max_requests: int, window_seconds: int) -> bool:
        # 基于时间窗口的限流逻辑
        ...
```

---

## 测试

```bash
# 运行单元测试
pytest tests/ -v

# 运行RAG测试
python test_rag.py

# 运行重排序测试
python test_rerank.py
```

### 测试覆盖

- 配置模块测试
- 异常类测试
- 日志配置测试
- 安全服务测试
- 统计模块测试

---

## 性能优化

### 已完成的优化

1. **统计模块异步写入**：后台定时刷新，避免频繁I/O
2. **Embedding缓存优化**：使用SHA256哈希作为缓存键
3. **API依赖注入**：移除全局单例，使用FastAPI依赖注入
4. **敏感词库扩展**：200+敏感词，分类管理
5. **关键词检索优化**：添加预过滤，减少全表扫描
6. **前端错误边界**：友好的错误提示
7. **配置安全解析**：无效值自动降级
8. **RAG引擎单例**：避免重复初始化
9. **SSE流式输出**：JSON格式+心跳+错误处理
10. **多轮对话优化**：历史长度控制
11. **API限流机制**：保护后端服务
12. **日志脱敏**：敏感信息保护
13. **两阶段检索**：初始召回 + 重排序，提升检索精度
14. **重排序模型集成**：使用BGE-Reranker-v2-m3 Cross-Encoder

---

## 注意事项

1. **HTTP代理问题**：项目已配置 `NO_PROXY` 环境变量，避免Ollama本地调用受代理影响
2. **Chroma缓存**：Streamlit使用 `@st.cache_resource` 缓存RAG引擎，注意重启应用以刷新状态
3. **Ollama服务**：确保Ollama服务已启动且模型已下载（包括重排序模型）
4. **API与Web共存**：可以同时运行Streamlit和FastAPI服务，端口不冲突
5. **限流配置**：生产环境建议启用限流并调整参数
6. **重排序模型**：首次使用需下载 `dengcao/bge-reranker-v2-m3:latest` 模型

---

## 项目版本

- v0.1.0 - 初始版本
- v0.1.1 - 性能优化版本
  - 添加异步统计写入
  - 添加API限流机制
  - 添加性能监控指标
  - 增强文档解析
  - 扩展测试覆盖
- v0.1.2 - 两阶段检索版本
  - 添加重排序模块（BGE-Reranker-v2-m3）
  - 支持两阶段检索：初始召回 → 精排
  - 添加重排序测试脚本
  - 知识库扩充（新增中国高血压防治指南2024）