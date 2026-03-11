"""前端常量定义"""

# API配置
DEFAULT_API_BASE_URL = "http://localhost:8000"

# 超时配置（秒）
DEFAULT_TIMEOUT = 30
STREAM_TIMEOUT = 300
UPLOAD_TIMEOUT = 600

# 缓存配置
STATS_CACHE_TIMEOUT = 30  # 统计缓存超时（秒）

# 输入限制
MAX_INPUT_CHARS = 500  # 输入字符数限制

# UI配置
CONTENT_TRUNCATE_LENGTH = 200  # 内容截断长度

# 知识库页面
MAX_FILE_SIZE_MB = 50  # 文件大小限制（MB）

# 统计页面
MAX_UNANSWERED_QUESTIONS = 20  # 未回答问题显示限制
