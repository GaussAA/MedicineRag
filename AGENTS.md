# 医疗知识问答系统 (MedicineRag)

## 项目概述

基于RAG（检索增强生成）技术的医疗知识问答系统，使用本地Ollama模型（Qwen3:8b + BGE-M3）进行问答，无需外接API，保护隐私。本系统为本科毕业设计项目，提供24小时可用的医疗咨询参考。

### 技术栈

| 层级         | 技术选型                  |
| ------------ | ------------------------- |
| 前端框架     | Streamlit 1.30+           |
| RAG框架      | LlamaIndex 0.10+         |
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
├── app/                    # Streamlit Web应用（前端）
│   ├── main.py             # 主问答页面
│   ├── api_client.py       # API客户端（前后端分离）
│   └── pages/
│       ├── knowledge.py    # 知识库管理页面
│       └── analytics.py    # 系统统计页面
├── backend/               # 后端服务
│   ├── config.py          # 配置管理（安全解析+限流配置）
│   ├── exceptions.py       # 自定义异常类
│   ├── logging_config.py  # 统一日志配置（请求ID追踪）
│   ├── statistics.py       # 问答统计模块（异步批量写入）
│   ├── api/                # FastAPI服务
│   │   ├── main.py         # API入口（限流中间件+健康检查）
│   │   ├── models.py       # 数据模型
│   │   ├── dependencies.py # 依赖注入
│   │   └── routes/         # API路由
│   │       ├── qa.py       # 问答API（依赖注入+流式输出）
│   │       └── docs.py     # 文档管理API
│   └── services/           # 服务层
│       ├── qa_service.py           # 问答服务（含查询缓存）
│       ├── doc_service.py         # 文档管理服务（含哈希缓存）
│       ├── security_service.py    # 安全检查服务
│       ├── question_type_detector.py  # 问题类型检测
│       └── confidence_calculator.py   # 置信度计算
├── rag/                    # RAG核心引擎
│   ├── engine.py           # RAG引擎实现（单例+LLM缓存+Embedding缓存）
│   ├── chunker.py          # 智能文档分块（含分块缓存）
│   ├── retriever.py        # 混合检索模块
│   ├── reranker.py         # 重排序模块（两阶段检索）
│   └── prompts.py          # Prompt模板
├── tests/                  # 单元测试（119个测试）
├── scripts/                # 启动脚本
│   ├── start_all.py        # 一键启动
│   └── stop_all.py         # 一键关闭
├── docs/                   # 项目文档
│   ├── spec.md            # 需求规格文档
│   └── technical-design.md # 技术方案设计
├── data/                   # 数据目录
│   ├── documents/          # 上传的源文档
│   ├── chroma_db/          # Chroma向量数据库
│   ├── embedding_cache/   # Embedding缓存
│   └── qa_stats.json       # 问答统计文件
├── storage/                # LlamaIndex存储（索引/图谱）
├── .env                    # 环境配置
├── pyproject.toml          # 项目配置
├── test_rag.py            # RAG测试脚本
├── test_rerank.py         # 重排序测试脚本
└── test_load.py           # 文档加载测试脚本
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
```

### 3. 运行应用

```bash
# 方式1：一键启动（推荐）
python scripts/start_all.py

# 方式2：分别启动
# 终端1：激活虚拟环境并运行Streamlit Web应用
source .venv/Scripts/activate
streamlit run app/main.py

# 终端2：运行API服务
source .venv/Scripts/activate
uvicorn backend.api.main:app --reload --port 8000

# 方式3：一键关闭
python scripts/stop_all.py
```

访问 http://localhost:8501 即可使用Web应用。

---

## 架构特点

### 前后端分离

本系统采用前后端分离架构：

- **前端**：Streamlit Web应用，通过HTTP API与后端通信
- **后端**：FastAPI服务，提供RESTful API
- **通信**：使用 `app/api_client.py` 统一封装API调用

这种架构的优势：
1. 前端后端独立开发、部署
2. 支持多客户端（Web、移动端）
3. 便于API复用和扩展
4. 更好的可测试性

### 依赖注入

后端API使用FastAPI依赖注入模式：

```python
# backend/api/dependencies.py
def get_rag_engine_dep() -> RAGEngine:
    """获取RAG引擎实例"""
    if not hasattr(get_rag_engine_dep, "_instance"):
        get_rag_engine_dep._instance = RAGEngine()
    return get_rag_engine_dep._instance

# backend/api/routes/qa.py
@router.post("/qa/ask")
async def ask(request: QARequest, qa_service: QAService = Depends(get_qa_service)):
    # 直接使用注入的服务
    response = qa_service.ask(qa_request)
```

---

## 核心功能

### 问答流程
1. 用户输入医疗问题
2. 安全检查（敏感词、紧急症状检测）
3. 问题类型自动检测（症状/疾病/用药/检查）
4. 向量检索（Chroma + BGE-M3）
5. **两阶段检索**：初始召回 → 重排序（BGE-Reranker-v2-m3）
6. 置信度计算与警告
7. LLM生成回答（Qwen3:8b）
8. 返回回答 + 参考来源 + 置信度警告 + 免责声明

### 知识库管理
- 支持PDF、Word、TXT、HTML、Markdown格式
- 智能分块、向量化存储
- 重建索引、清空知识库功能
- 文档列表、删除功能
- 重复文件检测（基于内容哈希）

### 高级功能

#### 两阶段检索（重排序）
- **第一阶段**：使用BGE-M3向量化检索，召回top_k=20个候选文档
- **第二阶段**：使用BGE-Reranker-v2-m3 Cross-Encoder对候选文档进行精排
- 返回top_k=5个最相关结果
- 可通过配置启用/禁用重排序功能

#### 问题类型检测
- 自动识别问题类型：症状相关、疾病相关、用药相关、检查相关
- 用于统计分析和优化回答策略

#### 置信度计算
- 基于检索结果的相关性分数计算置信度
- 低置信度时自动给出警告提示

#### 流式输出
- LLM回答逐字流式显示，减少等待焦虑
- SSE格式传输，支持心跳保持连接

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
- 知识库缺口识别（未回答问题追踪）
- 请求ID日志追踪

#### REST API
- 完整的RESTful API接口
- 支持文档上传、问答、统计等功能
- 自动生成API文档（Swagger UI）

#### 多层缓存机制
- **Embedding缓存**：LRU内存缓存 + 磁盘持久化，使用SHA256哈希作为缓存键
- **LLM响应缓存**：基于问题哈希的响应缓存，减少重复LLM调用
- **文档分块缓存**：基于内容哈希的分块结果缓存
- **查询分析缓存**：安全检查 + 问题类型检测结果缓存

---

## API接口

| 接口                    | 方法   | 功能         |
| ----------------------- | ------ | ------------ |
| `/api/qa/ask`           | POST   | 问答接口     |
| `/api/qa/stream`        | POST   | 流式问答     |
| `/api/docs/upload`      | POST   | 上传文档     |
| `/api/docs/list`        | GET    | 文档列表     |
| `/api/docs/{doc_id}`    | DELETE | 删除文档     |
| `/api/docs/rebuild`     | POST   | 重建索引     |
| `/api/docs/clear`       | POST   | 清空知识库   |
| `/api/docs/stats`       | GET    | 统计信息     |
| `/api/docs/stats/qa`    | GET    | 问答统计详情 |
| `/api/docs/stats/clear` | POST   | 清空统计数据 |
| `/health`               | GET    | 基础健康检查 |
| `/health/detailed`      | GET    | 详细健康检查 |
| `/metrics`              | GET    | 性能监控指标 |

访问 http://localhost:8000/docs 查看完整API文档。

---

## 配置说明

在 `.env` 文件中配置：

```bash
# Ollama配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=bge-m3:latest
OLLAMA_LLM_MODEL=qwen3:8b

# 重排序配置
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

# 向量数据库配置
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# 文档存储目录
DOCUMENTS_DIR=./data/documents

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./data/logs/app.log

# 缓存配置
ENABLE_EMBEDDING_CACHE=true
EMBEDDING_CACHE_SIZE=500

# 分块策略配置
CHUNK_BY_TITLE=true
PRESERVE_MEDICAL_TERMS=true

# API限流配置
ENABLE_RATE_LIMIT=true
RATE_LIMIT_QPS=10

# CORS配置（生产环境建议限制）
CORS_ALLOWED_ORIGINS=*

# 重复检测配置
ENABLE_DUPLICATE_CHECK=true
```

---

## 关键实现

### API客户端 (app/api_client.py)

前端通过API客户端与后端通信，含超时配置和Streamlit缓存：

```python
class APIClient:
    DEFAULT_TIMEOUT = 30
    STREAM_TIMEOUT = 300
    UPLOAD_TIMEOUT = 600
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        # 连接池配置...
    
    @st.cache_resource
    def get_api_client() -> APIClient:
        """获取API客户端单例（使用Streamlit缓存）"""
        return APIClient()
```

### RAG引擎 (rag/engine.py)

使用原生Ollama API避免HTTP代理连接池问题，采用单例模式，支持两阶段检索：

```python
class RAGEngine:
    _instance: Optional['RAGEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### Embedding缓存 (rag/engine.py)

LRU内存缓存 + 磁盘持久化，大幅减少重复计算：

```python
class EmbeddingCache:
    def __init__(self, max_size: int = 500, cache_dir: str = "data/embedding_cache"):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._cache_dir = Path(cache_dir)
        ...
```

### 统计模块 (backend/statistics.py)

异步批量写入，避免频繁I/O：

```python
class QAStats:
    def __init__(self, stats_file: str = "data/qa_stats.json"):
        self._lock = threading.Lock()
        self._flush_interval = 30  # 30秒刷新一次
        self._max_records_before_flush = 100
```

---

## 测试

```bash
# 运行单元测试（119个测试）
pytest tests/ -v

# 运行RAG测试
python test_rag.py

# 运行重排序测试
python test_rerank.py

# 运行文档加载测试
python test_load.py

# 查看测试覆盖率
pytest tests/ --cov=backend --cov=rag --cov=app
```

### 测试覆盖

- 配置模块测试
- 异常类测试
- 日志配置测试
- 安全服务测试
- 统计模块测试
- 分块器测试
- 检索器测试
- RAG核心测试
- 置信度计算器测试
- 问题类型检测器测试

---

## 性能优化

### 已完成的优化

#### 后端优化
1. **统计模块异步写入**：后台定时刷新，避免频繁I/O
2. **Embedding缓存优化**：LRU内存缓存 + 磁盘持久化，使用SHA256哈希作为缓存键
3. **API依赖注入**：移除全局单例，使用FastAPI依赖注入
4. **敏感词库扩展**：200+敏感词，分类管理
5. **关键词检索优化**：添加预过滤，减少全表扫描
6. **RAG引擎单例**：避免重复初始化
7. **SSE流式输出**：JSON格式+心跳+错误处理
8. **多轮对话优化**：历史长度控制
9. **API限流机制**：保护后端服务
10. **两阶段检索**：初始召回 + 重排序，提升检索精度
11. **问题类型检测**：自动识别问题类型
12. **置信度计算**：低置信度警告
13. **LLM响应缓存**：基于问题哈希的响应缓存
14. **文档分块缓存**：基于内容哈希的分块结果缓存
15. **查询分析缓存**：安全检查+问题类型检测结果缓存
16. **日志请求ID追踪**：基于ContextVar的请求级别日志
17. **降级处理**：LLM失败时返回检索到的资料
18. **哈希缓存优化**：文档哈希增量计算，避免重复读取
19. **移除重复导入**：将import语句移至文件顶部

#### 前端优化
1. **API客户端超时配置**：类常量统一管理超时参数
2. **Streamlit缓存**：@st.cache_resource缓存API客户端
3. **Stats缓存**：session_state缓存统计数据，30秒过期
4. **阻塞sleep替换**：手动刷新按钮替代time.sleep阻塞
5. **Bug修复**：sources变量初始化、建议按钮功能修复
6. **UX改进**：确认对话框、输入长度限制、成功提示

---

## 注意事项

1. **HTTP代理问题**：项目已配置 `NO_PROXY` 环境变量，避免Ollama本地调用受代理影响
2. **Chroma缓存**：Streamlit使用 `@st.cache_resource` 缓存RAG引擎，注意重启应用以刷新状态
3. **Ollama服务**：确保Ollama服务已启动且模型已下载（包括重排序模型）
4. **API与Web共存**：可以同时运行Streamlit和FastAPI服务，端口不冲突
5. **限流配置**：生产环境建议启用限流并调整参数
6. **重排序模型**：首次使用需下载 `dengcao/bge-reranker-v2-m3:latest` 模型
7. **启动顺序**：先启动后端API，再启动前端Streamlit
8. **日志目录**：首次运行前需创建 `./data/logs` 目录

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
- v0.1.3 - 前后端分离版本
  - 添加API客户端，实现真正的前后端分离
  - 添加问题类型检测
  - 添加置信度计算
  - 添加一键启动/关闭脚本
  - 添加Embedding缓存（LRU + 磁盘持久化）
  - 扩展测试覆盖至119个测试
  - 添加项目文档（需求规格 + 技术设计）
- v0.1.4 - 全面优化版本
  - 前端优化：超时配置、Stats缓存、阻塞sleep替换、Bug修复、UX改进
  - 后端优化：LLM响应缓存、文档分块缓存、查询分析缓存
  - 日志优化：请求ID追踪、降级处理
  - 代码优化：哈希缓存优化、移除重复导入
