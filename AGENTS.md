# 医疗知识问答系统 (MedicineRag)

## 项目概述

基于RAG（检索增强生成）技术的医疗知识问答系统，使用本地Ollama模型（Qwen3:8b + BGE-M3）进行问答，无需外接API，保护隐私。本系统支持两种问答模式：**Pipeline RAG**（传统流水线）和 **Agentic RAG**（智能体增强），采用ReAct推理模式实现更智能的问答体验。

### 技术栈

| 层级         | 技术选型                    |
| ------------ | --------------------------- |
| 前端框架     | Streamlit 1.30+             |
| RAG框架      | LlamaIndex 0.10+            |
| 向量数据库   | ChromaDB 0.4+               |
| LLM          | Ollama qwen3:8b             |
| Embedding    | BGE-M3:latest               |
| 重排序模型   | BGE-Reranker-v2-m3:latest   |
| Agent框架    | 自研ReAct + LangChain Tools |
| PDF解析      | pymupdf4llm (开源)          |
| API框架      | FastAPI 0.100+              |
| python包管理 | uv                          |

---

## 项目结构

```
MedicineRag/
├── app/                    # Streamlit Web应用（前端）
│   ├── main.py             # 主问答页面（支持Agent模式切换）
│   ├── api_client.py       # API客户端（支持Agent API + SSE解析优化）
│   ├── components.py       # 可复用UI组件（新增）
│   └── pages/
│       ├── knowledge.py    # 知识库管理页面
│       └── analytics.py    # 系统统计页面
│
├── backend/               # 后端服务
│   ├── config.py          # 配置管理（支持对话历史控制+日志优化配置）
│   ├── exceptions.py      # 自定义异常类
│   ├── logging_config.py  # 统一日志配置（结构化JSON+请求ID追踪）
│   ├── statistics.py      # 问答统计模块（增强统计指标+异步批量写入）
│   ├── api/              # FastAPI服务
│   │   ├── main.py       # API入口（限流中间件+健康检查）
│   │   ├── models.py     # 数据模型
│   │   ├── dependencies.py # 依赖注入（支持Agent）
│   │   └── routes/       # API路由
│   │       ├── qa.py     # 问答API + Agent API
│   │       └── docs.py   # 文档管理API
│   └── services/          # 服务层
│       ├── qa_service.py           # 问答服务（含查询缓存+公共方法重构）
│       ├── doc_service.py         # 文档管理服务（含哈希缓存）
│       ├── security_service.py    # 安全检查服务
│       ├── question_type_detector.py # 问题类型检测
│       └── confidence_calculator.py # 置信度计算
│
├── rag/                   # RAG核心引擎（模块化结构）
│   ├── __init__.py        # 包入口
│   ├── cache.py           # 缓存模块（统一接口+TTL支持）
│   ├── factory.py         # 组件工厂函数
│   ├── llm_manager.py     # LLM调用管理（配置化参数）
│   ├── vector_store.py    # 向量存储管理
│   ├── agents/            # Agent模块
│   │   ├── __init__.py
│   │   ├── base.py        # Agent抽象基类（ReAct框架）
│   │   ├── medical_agent.py # 医疗问答Agent实现
│   │   ├── tools/         # Agent工具集
│   │   │   ├── __init__.py
│   │   │   ├── retriever_tool.py    # 文档检索工具
│   │   │   ├── security_tool.py     # 安全检查工具
│   │   │   ├── knowledge_gap_tool.py # 知识缺口识别工具
│   │   │   └── followup_tool.py     # 主动追问工具
│   │   └── memory/        # 对话记忆
│   │       ├── __init__.py
│   │       └── conversation_memory.py # 对话记忆（支持Token裁剪）
│   ├── core/              # 核心模块
│   │   ├── engine.py      # RAG引擎实现（Facade模式+依赖注入）
│   │   ├── retriever.py   # 混合检索模块（含60+同义词扩展）
│   │   ├── reranker.py    # 重排序模块（两阶段检索）
│   │   └── prompts.py     # Prompt模板
│   └── processing/        # 处理模块
│       ├── chunker.py     # 智能文档分块（含分块缓存）
│       └── document_processor.py # 文档处理
│
├── tests/                 # 测试目录（195+个测试）
│   ├── conftest.py       # 共享fixtures
│   ├── unit/             # 单元测试
│   │   ├── test_config.py
│   │   ├── test_exceptions.py
│   │   ├── test_logging.py
│   │   └── test_statistics.py
│   ├── services/         # 服务层测试
│   │   ├── test_security.py
│   │   ├── test_question_type_detector.py
│   │   └── test_confidence_calculator.py
│   ├── rag/              # RAG核心测试 + Agent测试
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   ├── test_engine.py         # 含依赖注入测试
│   │   ├── test_cache.py
│   │   ├── test_agent_base.py      # Agent基础测试
│   │   ├── test_agent_tools.py    # Agent工具测试
│   │   ├── test_medical_agent.py  # 医疗Agent测试
│   │   └── test_conversation_memory.py # 对话记忆测试
│   ├── api/              # API测试（待扩展）
│   ├── integration/      # 集成测试（待扩展）
│   ├── fixtures/         # 测试数据
│   │   └── test_documents/
│   └── scripts/          # 测试脚本
│       ├── test_rag.py
│       ├── test_rerank.py
│       └── test_load.py
│
├── scripts/              # 启动脚本
│   ├── start_all.py      # 一键启动
│   └── stop_all.py       # 一键关闭
│
├── docs/                 # 项目文档
│   ├── spec.md           # 需求规格文档
│   └── technical-design.md # 技术方案设计
│
├── data/                 # 数据目录
│   ├── documents/        # 上传的源文档
│   ├── chroma_db/        # Chroma向量数据库
│   ├── embedding_cache/  # Embedding缓存
│   ├── logs/             # 日志目录
│   ├── storage/          # LlamaIndex存储
│   └── qa_stats.json    # 问答统计文件
│
├── .env                  # 环境配置
├── .env.example          # 环境配置模板
├── pyproject.toml        # 项目配置
├── AGENTS.md             # Agent说明文档
└── uv.lock               # 依赖锁定文件
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

# 终端2：激活虚拟系统并运行API服务
source .venv/Scripts/activate
uvicorn backend.api.main:app --reload --port 8000

# 方式3：一键关闭
python scripts/stop_all.py
```

访问 http://localhost:8501 即可使用Web应用。

---

## 架构特点

### Agentic RAG (v0.3.0+)

本系统支持两种问答模式：

#### Pipeline RAG（传统流水线）
- 线性流程：问题 → 安全检查 → 检索 → 重排序 → LLM生成
- 简单直接，适合标准问答场景

#### Agentic RAG（智能体增强）
- **ReAct推理模式（优化版）**：思考+决策+反思合并为单次LLM调用，大幅提升响应速度
- **工具调用**：Agent自主决定调用哪些工具
- **动态决策**：根据检索结果决定是否需要补充检索
- **知识缺口识别**：自动识别知识库薄弱领域
- **对话历史增强**：自动获取最近3轮对话上下文

```
用户问题 → Agent(优化ReAct循环)
           ├── 步骤N: 思考+决策+反思（一次LLM调用）
           │         └── 分析意图 → 选择工具 → 判断是否继续
           ├── 执行行动：调用工具（检索/安全检查/追问/知识缺口）
           └── 观察结果 → 更新上下文
           → 生成最终回答
```

### 模块化设计

RAG核心采用Facade模式和依赖注入：

- **rag/core/**: 核心算法（Engine, Retriever, Reranker, Prompts）
- **rag/processing/**: 处理流程（Chunker, DocumentProcessor）
- **rag/agents/**: Agent框架（BaseAgent, MedicalAgent, Tools）
- **rag/cache.py**: 统一缓存接口（支持TTL）
- **rag/factory.py**: 组件工厂函数
- **rag/llm_manager.py**: LLM调用管理
- **rag/vector_store.py**: 向量存储管理

### 前后端分离

本系统采用前后端分离架构：

- **前端**：Streamlit Web应用，通过HTTP API与后端通信
- **后端**：FastAPI服务，提供RESTful API
- **通信**：使用 `app/api_client.py` 统一封装API调用

---

## 核心功能

### 问答流程

#### Pipeline RAG 模式
1. 用户输入医疗问题
2. 安全检查（敏感词、紧急症状检测）
3. 问题类型自动检测（症状/疾病/用药/检查）
4. 向量检索（Chroma + BGE-M3）
5. **两阶段检索**：初始召回 → 重排序（BGE-Reranker-v2-m3）
6. **相似度阈值过滤**：过滤低相关性文档（SIMILARITY_THRESHOLD）
7. 置信度计算与警告
8. LLM生成回答（Qwen3:8b）
9. 返回回答 + 参考来源 + 置信度警告 + 免责声明

#### Agentic RAG 模式
1. 用户输入医疗问题
2. Agent接收问题，进入ReAct推理循环
3. **思考步骤**：LLM分析问题，决定下一步行动
4. **行动执行**：调用工具（可多轮）
   - 安全检查工具：检测敏感内容/紧急症状
   - 检索工具：查询知识库
   - 追问工具：生成跟进问题
   - 知识缺口工具：识别知识库薄弱领域
5. **反思步骤**：判断结果是否足够
6. 生成最终回答
7. 返回回答 + 推理步骤 + 参考来源 + 置信度 + 追问问题 + 知识缺口

### Agent工具集

| 工具                        | 功能         | 说明                       |
| --------------------------- | ------------ | -------------------------- |
| retrieve_docs               | 文档检索     | 调用RAG引擎检索相关文档    |
| check_security              | 安全检查     | 检测敏感内容和紧急医疗情况 |
| generate_followup_questions | 主动追问     | 基于问题和答案生成跟进问题 |
| identify_knowledge_gap      | 知识缺口识别 | 识别知识库薄弱领域         |

### 知识库管理
- 支持PDF、Word、TXT、HTML、Markdown格式
- **PDF解析**：使用pymupdf4llm（免费开源，输出Markdown格式）
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

#### 相似度阈值过滤
- 根据 `SIMILARITY_THRESHOLD` 配置过滤低相关性文档
- 默认阈值0.3，低于此分数的文档将被过滤

#### 查询扩展
- 60+组医学同义词扩展（心血管，呼吸、消化、泌尿、内分泌、神经、骨科、皮肤、眼科等）
- 自动拼写错误纠正
- 提升检索召回率

#### 问题类型检测
- 自动识别问题类型：症状相关、疾病相关、用药相关、检查相关
- **中英文双语支持**：150+关键词（中英文）
- 用于统计分析和优化回答策略

#### 置信度计算
- **多维度加权计算**：
  - 文档数量得分 (0-0.2)
  - 相似度分数 (0-0.4)
  - 问题覆盖度 (0-0.4)
  - 来源标注加分 (0-0.1)
- **始终显示知识库匹配度百分比**（高/中/低三种级别）
- 低置信度时自动给出警告提示

| 匹配度 | 显示                                    |
| ------ | --------------------------------------- |
| >= 75% | ✅ 知识库匹配度良好（XX%）               |
| 60-74% | ℹ️ 知识库匹配度一般（XX%）               |
| < 60%  | ⚠️ 知识库匹配度较低（XX%），回答仅供参考 |

#### 流式输出
- LLM回答逐字流式显示，减少等待焦虑
- SSE格式传输，支持心跳保持连接

#### 多轮对话
- 支持上下文理解（可配置轮数，默认5轮）
- 历史长度自动控制，避免上下文溢出
- 对话记忆持久化管理
- **Token自动裁剪**：基于字符近似估算（约1.5字符/token）

#### API限流保护
- 问答接口：30次/分钟
- 文档上传：10次/分钟
- Agent接口：20次/分钟
- 其他接口：60次/分钟
- 可通过环境变量配置

#### 可观测性
- 问答统计面板（增强指标）
- 问题类型分布分析
- 性能指标监控（/metrics端点）
- 知识库缺口识别（未回答问题追踪）
- 请求ID日志追踪
- **结构化JSON日志**（可选，方便日志分析系统）

#### REST API
- 完整的RESTful API接口
- 支持文档上传、问答、Agent问答等功能
- 自动生成API文档（Swagger UI）

#### 多层缓存机制
- **Embedding缓存**：LRU内存缓存 + 磁盘持久化，使用SHA256哈希作为缓存键
- **LLM响应缓存**：基于问题哈希的响应缓存，减少重复LLM调用
- **文档分块缓存**：基于内容哈希的分块结果缓存（类实例变量）
- **查询分析缓存**：安全检查 + 问题类型检测结果缓存

---

## API接口

| 接口                    | 方法   | 功能             |
| ----------------------- | ------ | ---------------- |
| `/api/qa/ask`           | POST   | Pipeline问答     |
| `/api/qa/stream`        | POST   | Pipeline流式问答 |
| `/api/qa/agent`         | POST   | Agent问答        |
| `/api/qa/agent/stream`  | POST   | Agent流式问答    |
| `/api/docs/upload`      | POST   | 上传文档         |
| `/api/docs/list`        | GET    | 文档列表         |
| `/api/docs/{doc_id}`    | DELETE | 删除文档         |
| `/api/docs/rebuild`     | POST   | 重建索引         |
| `/api/docs/clear`       | POST   | 清空知识库       |
| `/api/docs/stats`       | GET    | 统计信息         |
| `/api/docs/stats/qa`    | GET    | 问答统计详情     |
| `/api/docs/stats/clear` | POST   | 清空统计数据     |
| `/health`               | GET    | 基础健康检查     |
| `/health/detailed`      | GET    | 详细健康检查     |
| `/metrics`              | GET    | 性能监控指标     |

访问 http://localhost:8000/docs 查看完整API文档。

---

## 配置说明

在 `.env` 文件中配置（完整配置项见 `.env.example`）：

```bash
# ============================================================================
# Ollama配置
# ============================================================================
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=bge-m3:latest
OLLAMA_LLM_MODEL=qwen3:8b
OLLAMA_RERANK_MODEL=dengcao/bge-reranker-v2-m3:latest

# ============================================================================
# 重排序配置
# ============================================================================
ENABLE_RERANK=true
RERANK_INITIAL_TOP_K=20

# ============================================================================
# RAG配置 - 文档分块参数
# ============================================================================
CHUNK_SIZE=512
CHUNK_OVERLAP=50
CHUNK_MAX_LENGTH=2000

# ============================================================================
# RAG配置 - 检索参数
# ============================================================================
TOP_K=5
SIMILARITY_THRESHOLD=0.3

# ============================================================================
# LLM生成参数
# ============================================================================
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=512

# ============================================================================
# 对话历史控制参数（v0.3.1+ 新增）
# ============================================================================
MAX_HISTORY_TURNS=5
MAX_ANSWER_LENGTH=300
MAX_CONTEXT_LENGTH=1500

# ============================================================================
# 日志配置（v0.3.1+ 增强）
# ============================================================================
LOG_LEVEL=INFO
LOG_FILE=./data/logs/app.log
USE_STRUCTURED_LOG=false          # 是否使用结构化JSON日志
LOG_MAX_BYTES=10485760          # 日志文件最大字节数（默认10MB）
LOG_BACKUP_COUNT=5              # 日志备份文件数量

# ============================================================================
# 缓存配置
# ============================================================================
ENABLE_EMBEDDING_CACHE=true
EMBEDDING_CACHE_SIZE=500

# ============================================================================
# 分块策略配置
# ============================================================================
CHUNK_BY_TITLE=true
PRESERVE_MEDICAL_TERMS=true

# ============================================================================
# API限流配置
# ============================================================================
ENABLE_RATE_LIMIT=true
RATE_LIMIT_QPS=10

# ============================================================================
# CORS配置（生产环境建议限制）
# ============================================================================
CORS_ALLOWED_ORIGINS=*

# ============================================================================
# 重复检测配置
# ============================================================================
ENABLE_DUPLICATE_CHECK=true
```

---

## 关键实现

### 统一缓存接口 (rag/cache.py)

```python
class BaseCache(ABC):
    """缓存抽象基类"""
    @abstractmethod
    def get(self, key: str) -> Optional[Any]: ...
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None): ...
    @abstractmethod
    def clear(self): ...

class TTLCache(BaseCache):
    """支持TTL的缓存实现"""
    def __init__(self, max_size: int = 100, default_ttl: int = 3600): ...
```

### RAGEngine依赖注入 (rag/core/engine.py)

```python
class RAGEngine:
    @classmethod
    def create_instance(cls, **kwargs) -> 'RAGEngine':
        """创建新的RAGEngine实例（非单例）"""
        ...
    
    def get_component(self, component_name: str):
        """获取内部组件"""
        ...
```

### QAService公共方法 (backend/services/qa_service.py)

```python
class QAService:
    def _analyze_query(self, question: str) -> Dict[str, Any]:
        """分析查询：问题类型检测 + 安全检查 + 紧急症状检查"""
        ...
    
    def _build_response(self, question: str, ...) -> QAResponse:
        """构建问答响应（统一响应生成逻辑）"""
        ...
```

### API客户端SSE解析 (app/api_client.py)

```python
class APIClient:
    def _parse_sse_stream(self, response, ...) -> Generator[str, None, None]:
        """统一SSE流式响应解析"""
        ...
    
    def _handle_request_error(self, error: Exception) -> str:
        """统一请求错误处理"""
        ...
```

### 对话记忆Token裁剪 (rag/memory/conversation_memory.py)

```python
class ConversationMemory:
    def _trim_by_token_estimate(self, context: "ConversationContext") -> None:
        """根据token估算值裁剪上下文（约1.5字符/token）"""
        ...
```

---

## 测试

```bash
# 运行所有测试（195+个测试）
pytest tests/ -v

# 运行Agent相关测试
pytest tests/rag/test_agent_base.py -v       # Agent基础测试
pytest tests/rag/test_agent_tools.py -v       # Agent工具测试
pytest tests/rag/test_medical_agent.py -v     # 医疗Agent测试
pytest tests/rag/test_conversation_memory.py -v # 对话记忆测试

# 运行特定目录的测试
pytest tests/unit/ -v          # 单元测试
pytest tests/services/ -v     # 服务层测试
pytest tests/rag/ -v          # RAG核心测试

# 运行测试脚本
python tests/scripts/test_rag.py
python tests/scripts/test_rerank.py
python tests/scripts/test_load.py
```

### 测试覆盖

- **单元测试** (tests/unit/): 配置、异常、日志、统计
- **服务层测试** (tests/services/): 安全服务、问题类型检测、置信度计算
- **RAG核心测试** (tests/rag/): 分块器、检索器、引擎、缓存、依赖注入
- **Agent测试** (tests/rag/): BaseAgent、工具、对话记忆、医疗Agent
- **API测试** (tests/api/): 待扩展
- **集成测试** (tests/integration/): 待扩展

新增测试用例：
- `TestRAGEngineDependencyInjection`: RAG引擎依赖注入测试
  - create_instance基本功能测试
  - 独立于单例的实例创建测试
  - 自定义组件注入测试
  - get_component方法测试

---

## 性能优化

### 已完成的优化

#### Agent优化
1. **ReAct推理循环**：思考→行动→观察→反思的智能决策
2. **工具调用**：Agent自主选择调用安全检查/检索/追问等工具
3. **反思机制**：基于检索结果判断是否需要继续
4. **步骤缓存优化**：避免LLM响应缓存冲突
5. **Context更新修复**：确保每步后更新上下文状态
6. **合并思考-决策-反思**：从每步2-3次LLM调用减少到1次，大幅提升速度
7. **对话历史增强**：自动获取最近3轮对话上下文
8. **多维度置信度计算**：文档数量+相似度+问题覆盖度+来源标注
9. **异步执行支持**：添加arun异步执行方法
10. **降级策略**：LLM调用失败时使用默认逻辑

#### 后端优化
1. **统计模块异步写入**：后台定时刷新，避免频繁I/O
2. **Embedding缓存优化**：LRU内存缓存 + 磁盘持久化
3. **API依赖注入**：移除全局单例，使用FastAPI依赖注入
4. **敏感词库扩展**：200+敏感词，分类管理
5. **关键词检索优化**：添加预过滤，减少全表扫描
6. **RAG引擎单例**：避免重复初始化
7. **SSE流式输出**：JSON格式+心跳+错误处理
8. **多轮对话优化**：历史长度控制
9. **API限流机制**：保护后端服务
10. **两阶段检索**：初始召回 + 重排序，提升检索精度
11. **问题类型检测**：自动识别问题类型（中英文双语支持）
12. **置信度计算**：始终显示知识库匹配度百分比
13. **LLM响应缓存**：基于问题哈希的响应缓存
14. **文档分块缓存**：基于内容哈希的分块结果缓存
15. **查询分析缓存**：安全检查+问题类型检测结果缓存
16. **日志请求ID追踪**：基于ContextVar的请求级别日志
17. **降级处理**：LLM失败时返回检索到的资料
18. **相似度阈值过滤**：实际应用SIMILARITY_THRESHOLD配置
19. **同义词扩展**：60+组医学同义词词典
20. **LLM参数优化**：TEMPERATURE=0.1, MAX_TOKENS=512，提升响应速度
21. **缓存统计记录**：自动记录LLM缓存命中/未命中统计
22. **统一缓存接口**：抽象基类+TTL支持+统一持久化
23. **RAGEngine依赖注入**：支持非单例创建和自定义组件注入
24. **QAService重构**：提取公共方法(_analyze_query, _build_response)减少重复代码
25. **API错误处理完善**：更详细的HTTP状态码和错误信息

#### 前端优化
1. **API客户端超时配置**：类常量统一管理超时参数
2. **Streamlit缓存**：@st.cache_resource缓存API客户端
3. **Stats缓存**：session_state缓存统计数据，30秒过期
4. **Agent模式切换**：UI支持Pipeline/Agent模式切换
5. **知识库状态同步**：删除/清空知识库后自动刷新缓存
6. **UI组件模块化**：提取可复用组件(app/components.py)
7. **SSE解析优化**：统一解析方法支持多种消息类型

#### 日志系统优化 (v0.3.1+)
1. **结构化JSON日志**：支持导出为JSON格式，方便ELK/Graylog分析
2. **日志轮转配置化**：LOG_MAX_BYTES、LOG_BACKUP_COUNT可配置
3. **请求级别追踪**：基于ContextVar的请求ID追踪
4. **敏感信息脱敏**：自动过滤身份证、手机号、银行卡、邮箱

#### 增强统计指标 (v0.3.1+)
1. **每小时问答量**：按小时统计问答数量
2. **响应时间百分位数**：P50/P95/P99延迟统计
3. **最大/最小响应时间**：记录响应时间范围
4. **对话Token裁剪**：基于字符近似估算自动裁剪

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
9. **PDF解析**：默认使用 pymupdf4llm（免费开源）
10. **Agent模式**：在前端界面可切换Pipeline RAG / Agentic RAG 模式
11. **结构化日志**：设置 `USE_STRUCTURED_LOG=true` 启用JSON格式日志

---

## 项目版本

- v0.1.0 - 初始版本
- v0.1.1 - 性能优化版本
- v0.1.2 - 两阶段检索版本
- v0.1.3 - 前后端分离版本
- v0.1.4 - 全面优化版本
- v0.1.5 - 稳定性修复版本
- v0.1.6 - 置信度计算修复版本
- v0.2.0 - 模块化重构版本
- v0.3.0 - **Agentic RAG版本**
  - 添加Agent框架（ReAct推理模式）
  - 添加Agent工具集（检索/安全/追问/知识缺口）
- v0.3.1 - **工程化重构版本**
  - 统一缓存接口（TTL支持）
  - RAGEngine依赖注入改造
  - QAService公共方法提取
  - 前端组件模块化（app/components.py）
  - API客户端SSE解析优化
  - 完善的错误处理和测试用例
  - 添加对话记忆模块（含Token裁剪）
  - 添加Agent API端点
  - 扩展测试至195+个
  - 前端支持Agent模式切换
- v0.3.2 - **日志系统优化版本**
  - 结构化JSON日志支持
  - 日志轮转配置化（LOG_MAX_BYTES, LOG_BACKUP_COUNT）
  - 对话历史控制参数（MAX_HISTORY_TURNS等）
  - 增强统计指标（每小时问答量、P50/P95/P99延迟）
  - 完善.env.example配置文件