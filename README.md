# RAG Agent API (Production)

生产级 RAG Agent API 服务，支持 API Key 鉴权、速率限制、结构化日志、Docker 部署。

## 快速开始

### 本地运行

```powershell
cd rag-agent-api
python rag_api.py
```

### Docker 运行

```powershell
# 构建并启动
docker compose up -d --build

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

## 环境变量

在 `.env` 中配置（已自动读取）：

| 变量 | 说明 | 默认值 |
|---|---|---|
| DEEPSEEK_API_KEY | DeepSeek API Key | 必填 |
| RAG_API_KEY | API 鉴权密钥 | rag-secret-key-2024 |
| RAG_RATE_LIMIT | 每分钟最大请求数 | 30 |

## 功能

| 功能 | 说明 | 版本 |
|---|---|---|
| API Key 鉴权 | X-API-Key 请求头校验，无 Key 返回 401 | v1.0 |
| 速率限制 | 滑动窗口限流，超限返回 429 | v1.1 |
| 结构化日志 | 全链路 trace_id 追踪，单行 JSON 格式，自动写入文件 | v1.2 |
| Docker 部署 | 容器化运行，持久化 chroma_db + 日志 + 模型缓存 | v1.3 |
| 多路召回 | 稠密向量 + BM25 稀疏 + RRF 融合 | v2.0 |
| Reranker 精排 | Cross-Encoder 重排序提升精度 | v2.0 |
| Multi-Agent 编排 | 研究员、写作者、审核员，带持久化记忆和监控 | v2.0 |

## API

| 接口 | 鉴权 | 说明 |
|---|---|---|
| GET /health | 免鉴权 | 健康检查 |
| POST /query | X-API-Key | 标准问答（FC 驱动 RAG） |
| POST /query/stream | X-API-Key | 流式问答（SSE） |
| POST /query/hybrid | X-API-Key | 多路召回（稠密 + BM25），可选 Reranker |
| POST /doc | X-API-Key | 添加知识 |
| GET /kb/docs | X-API-Key | 获取知识库全部文档 |
| POST /agent/write | X-API-Key | Multi-Agent 写作流水线 |

### 测试命令

```powershell
# 不带 Key → 401
$body = '{"question":"What is Python?"}'
curl.exe -X POST http://localhost:8000/query -H "Content-Type: application/json" -d $body

# 带 Key → 200
curl.exe -X POST http://localhost:8000/query -H "Content-Type: application/json" -H "X-API-Key: rag-secret-key-2024" -d $body

# 自定义 trace_id
curl.exe -X POST http://localhost:8000/query -H "Content-Type: application/json" -H "X-API-Key: rag-secret-key-2024" -H "X-Trace-Id: my-id-001" -d $body

# 流式
$body = '{"question":"What is PyTorch? Summarize in Chinese"}'
curl.exe -N -X POST http://localhost:8000/query/stream -H "Content-Type: application/json" -H "X-API-Key: rag-secret-key-2024" -d $body

# 添加知识
$body = '{"title":"FastAPI","content":"FastAPI is a modern Python web framework."}'
curl.exe -X POST http://localhost:8000/doc -H "Content-Type: application/json" -H "X-API-Key: rag-secret-key-2024" -d $body
```

## 日志

日志写入 `rag_api.log`，每条请求带 trace_id：

```
2026-07-15 02:50:15 | INFO  | rag-api | [0be563ab] | tool_call | tool=search_knowledge | args={"query": "PyTorch"} | result=PyTorch was developed...
2026-07-15 02:50:21 | INFO  | rag-api | [0be563ab] | response | status=200 | duration=5.914s
```

实时监控：

```powershell
Get-Content rag_api.log -Tail 5 -Wait
```
