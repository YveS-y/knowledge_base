# RagStudy 项目说明

## 项目概述

RagStudy 是一个基于 RAG（检索增强生成）的知识库系统，用 Python 3.11+ 编写。核心功能：将文档（PDF/Markdown）导入向量数据库，再通过 LLM 对话检索回答。

## 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 工作流引擎 | LangGraph |
| LLM 接入 | LangChain + OpenAI 兼容 API（Qwen3/DashScope） |
| 向量模型 | BGE-M3（dense + sparse 混合向量） |
| 重排序 | BGE-Reranker |
| PDF 解析 | MineRU API + Magic-PDF |
| 向量数据库 | Milvus |
| 文档数据库 | MongoDB（对话历史） |
| 对象存储 | MinIO（文件/图片） |
| 知识图谱 | Neo4j |
| 日志 | Loguru |
| 配置管理 | python-dotenv + dataclass |

## 目录结构

```
RagStudy/
├── app/
│   ├── clients/           # 数据库客户端
│   │   ├── milvus_utils.py
│   │   ├── minio_utils.py
│   │   ├── mongo_history_utils.py
│   │   ├── mongo_history_utils_new.py
│   │   └── neo4j_utils.py
│   ├── conf/              # 配置（从 .env 读取，dataclass 单例）
│   │   ├── lm_config.py
│   │   ├── embedding_config.py
│   │   ├── reranker_config.py
│   │   ├── milvus_config.py
│   │   ├── minio_config.py
│   │   ├── mineru_config.py
│   │   └── bailian_mcp_config.py
│   ├── core/
│   │   ├── logger.py      # loguru 日志，自动定位真实调用位置
│   │   └── load_prompt.py # 提示词模板加载
│   ├── lm/
│   │   ├── lm_utils.py        # LLM 客户端（带缓存）
│   │   ├── embedding_utils.py # BGE-M3 向量生成（单例）
│   │   └── reranker_utils.py  # BGE 重排序
│   ├── utils/
│   │   ├── sse_utils.py       # SSE 流式响应（READY/PROGRESS/DELTA/FINAL/ERROR/CLOSE）
│   │   ├── task_utils.py      # 任务进度追踪（内存状态 + SSE 集成）
│   │   ├── format_utils.py    # JSON 格式化（支持中文）
│   │   ├── path_util.py       # 项目根路径检测（通过 .env 文件定位）
│   │   ├── rate_limit_utils.py         # 滑动窗口限速
│   │   ├── escape_milvus_string_utils.py # Milvus 字符串转义
│   │   └── normalize_sparse_vector.py  # 稀疏向量 L2 归一化
│   ├── tool/
│   │   ├── download_bgem3.py      # 下载 BGE-M3 模型
│   │   └── download_reranker.py   # 下载 BGE Reranker 模型
│   └── import_process/        # 文档导入工作流（LangGraph）
│       └── agent/
│           ├── state.py       # 图状态定义（ImportGraphState TypedDict）
│           └── nodes/         # 工作流节点（待实现）
├── doc/                   # 示例文档（中文产品手册 PDF）
├── test/                  # 测试脚本
├── logs/                  # 日志文件（自动生成，按天滚动）
├── pyproject.toml
└── .env                   # 密钥和配置（不提交到 git）
```

## 主要工作流

### 文档导入流程（LangGraph）
```
PDF/Markdown 文件
  → 1. PDF → Markdown（MineRU API）
  → 2. 提取 item_name（产品名识别）
  → 3. 文档切块（normal split / advanced split）
  → 4. 生成向量（BGE-M3 dense + sparse）
  → 5. 写入 Milvus
  → （可选）写入 Neo4j 知识图谱
```

### 查询流程
```
用户提问
  → Milvus 混合检索（dense + sparse）
  → BGE-Reranker 重排序
  → （可选）MCP Web 搜索 / Neo4j 图查询
  → LLM 生成回答（SSE 流式输出）
  → 保存对话历史到 MongoDB
```

## 关键设计模式

- **单例模式**：数据库客户端、LLM 客户端、BGE-M3 模型均为单例，避免重复初始化
- **缓存**：LLM 客户端按 `(model, json_mode)` 缓存
- **状态机**：LangGraph 管理多步骤导入流程，状态定义在 `ImportGraphState`
- **SSE 流式**：导入进度和 LLM 回答均通过 SSE 推送给前端
- **配置优先级**：系统环境变量 > `.env` 文件 > 代码默认值

## 外部服务

| 服务 | 地址 | 用途 |
|------|------|------|
| Milvus | http://47.94.86.115:19530 | 向量存储与检索 |
| MongoDB | mongodb://47.94.86.115:27017 | 对话历史 |
| MinIO | http://47.94.86.115:9000 | 文件/图片存储 |
| Neo4j | bolt://192.168.11.104:7687 | 知识图谱 |
| DashScope | https://dashscope.aliyuncs.com | LLM API（Qwen3） |
| MineRU | https://mineru.net/api/v4 | PDF 解析 |

## 日志

- 路径：`logs/app_YYYYMMDD.log`
- 格式：`[时间] | LEVEL | 文件名:函数:行号 - 消息`
- 支持中文，自动滚动，按 `.env` 配置保留天数

## 当前进度

- 基础设施层（配置、日志、客户端、工具）已完成
- `import_process/agent/state.py` 已定义图状态
- `import_process/agent/nodes/` 节点尚未实现
- FastAPI 入口文件（main.py）尚未创建
