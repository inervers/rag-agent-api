"""L5-Day3: FC RAG API + 结构化日志（API Key + 限流 + 全链路追踪）

- 鉴权：X-API-Key 请求头
- 限流：滑动窗口（配置 RAG_RATE_LIMIT）
- 日志：trace_id 追踪、工具调用链路、请求耗时、错误记录
"""

import sys, os, json

# Windows user-site 兼容（Docker 中直接跳过）
_REAL_USER_SITE = os.environ.get("PYTHON_USER_SITE")
if _REAL_USER_SITE and os.path.isdir(_REAL_USER_SITE) and _REAL_USER_SITE not in sys.path:
    sys.path.insert(0, _REAL_USER_SITE)

os.environ.setdefault("HF_HOME", r"C:\Users\inervers\Desktop\OH-WorkSpace\dl-learning\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# =============================================
# 配置加载（从 .env + 环境变量）
# =============================================

CONFIG = {}

def _load_config():
    search_dir = os.path.dirname(__file__)
    for _ in range(6):
        env_path = os.path.join(search_dir, ".env")
        if os.path.isfile(env_path):
            for line in open(env_path, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    CONFIG[k] = v
                    os.environ.setdefault(k, v)
            break
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent

_load_config()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
RAG_API_KEY = os.environ.get("RAG_API_KEY", "rag-secret-key-2024")
RATE_LIMIT = int(os.environ.get("RAG_RATE_LIMIT", "30"))

if not DEEPSEEK_API_KEY:
    print("需要设置 DEEPSEEK_API_KEY")
    exit(1)

import time, uuid, logging, traceback
import httpx
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import chromadb
from chromadb.api.types import EmbeddingFunction

# =============================================
# 结构化日志（终端 + 文件）
# =============================================

LOG_FILE = os.environ.get("RAG_LOG_FILE", os.path.join(os.path.dirname(__file__), "rag_api.log"))

logger = logging.getLogger("rag-api")
logger.setLevel(logging.INFO)

# 终端 handler（stderr）
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', datefmt='%H:%M:%S'))
logger.addHandler(console)

# 文件 handler（追加，UTF-8）
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-5s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(file_handler)

def _log(trace_id: str, event: str, **fields):
    """结构化日志：一行一个事件，json 字段便于 grep"""
    parts = [f"[{trace_id[:8]}]", event]
    for k, v in fields.items():
        parts.append(f"{k}={v}")
    logger.info(" | ".join(parts))

def _log_tool(trace_id: str, func_name: str, args: dict, result_preview: str):
    _log(trace_id, "tool_call", tool=func_name, args=json.dumps(args, ensure_ascii=False), result=result_preview[:80])

# =============================================
# 速率限制器
# =============================================

WINDOW_SEC = 60

class RateLimiter:
    def __init__(self):
        self._records: dict[str, list[float]] = {}

    def check(self, key: str) -> tuple[bool, int, int]:
        now = time.time()
        window_start = now - WINDOW_SEC
        if key not in self._records:
            self._records[key] = []
        self._records[key] = [t for t in self._records[key] if t > window_start]
        used = len(self._records[key])
        if used >= RATE_LIMIT:
            return False, used, RATE_LIMIT
        self._records[key].append(now)
        return True, used + 1, RATE_LIMIT

rate_limiter = RateLimiter()

# =============================================
# 统一中间件（鉴权 + 限流 + 日志追踪）
# =============================================

AUTH_HEADER = "X-API-Key"
TRACE_HEADER = "X-Trace-Id"

async def logging_middleware(request: Request, call_next):
    """最外层中间件：追踪、耗时、错误记录"""
    trace_id = request.headers.get(TRACE_HEADER, uuid.uuid4().hex)
    start = time.time()

    _log(trace_id, "request", method=request.method, path=request.url.path)

    try:
        response = await call_next(request)
        duration = round(time.time() - start, 3)
        response.headers["X-Trace-Id"] = trace_id
        if response.status_code < 400:
            _log(trace_id, "response", status=response.status_code, duration=f"{duration}s")
        else:
            _log(trace_id, "response_error", status=response.status_code, duration=f"{duration}s")
        return response
    except Exception as e:
        duration = round(time.time() - start, 3)
        tb = traceback.format_exc()
        _log(trace_id, "unhandled_error", error=str(e), duration=f"{duration}s")
        logger.error(f"[{trace_id[:8]}] Unhandled:\n{tb}")
        return JSONResponse(status_code=500, content={"error": "Internal server error", "trace_id": trace_id})


async def security_middleware(request: Request, call_next):
    """安全检查（鉴权 → 限流）"""
    if request.url.path == "/health":
        return await call_next(request)

    trace_id = request.headers.get(TRACE_HEADER, uuid.uuid4().hex)

    # 鉴权
    api_key = request.headers.get(AUTH_HEADER)
    if not api_key:
        _log(trace_id, "auth_failed", reason="missing_key")
        return JSONResponse(status_code=401, content={"error": "Missing X-API-Key header"})
    if api_key != RAG_API_KEY:
        _log(trace_id, "auth_failed", reason="invalid_key")
        return JSONResponse(status_code=403, content={"error": "Invalid API Key"})

    # 限流
    allowed, used, limit = rate_limiter.check(api_key)
    if not allowed:
        _log(trace_id, "rate_limited", used=used, limit=limit)
        return JSONResponse(
            status_code=429,
            content={"error": f"Rate limit exceeded: {used}/{limit} per minute"},
            headers={"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"},
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(limit - used)
    return response

# =============================================
# 嵌入模型
# =============================================
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

def embed_texts(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=256)
    with torch.no_grad():
        pooled = model(**inputs).last_hidden_state.mean(dim=1)
    return (pooled / torch.norm(pooled, dim=1, keepdim=True)).numpy()

class MiniLMEmbedding(EmbeddingFunction):
    def __call__(self, texts):
        return embed_texts(texts).tolist()

# =============================================
# Chroma 持久化
# =============================================
CHROMA_DIR = os.environ.get("RAG_CHROMA_DIR", os.path.join(os.path.dirname(__file__), "chroma_db"))
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=chromadb.config.Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="rag_knowledge", embedding_function=MiniLMEmbedding())
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

def _doc_count() -> int:
    try:
        return collection.count()
    except:
        return 0

def _doc_ids(start: int, n: int) -> list[str]:
    return [f"doc_{start + i}" for i in range(n)]

if _doc_count() == 0:
    init_texts = [
        "Python was created by Guido van Rossum and first released in 1991. It is a high-level general-purpose programming language emphasizing code readability with significant indentation.",
        "PyTorch was developed by Meta AI (Facebook AI Research) and released in 2016. Key features include dynamic computation graphs, GPU-accelerated tensor computation, automatic differentiation with Autograd.",
        "The Transformer architecture was introduced by Google in the 2017 paper 'Attention Is All You Need'. It is the foundation for BERT, GPT, T5, and ViT.",
        "RAG (Retrieval-Augmented Generation) combines a retriever and a generator. The retriever searches a knowledge base for relevant documents to produce informed answers.",
        "Chroma is an open-source vector database built for AI applications. It supports persistent storage and integrates natively with LangChain and LlamaIndex.",
        "LangChain is an open-source framework for LLM application development. It provides modular abstractions for models, prompts, chains, memory, agents, and retrieval.",
    ]
    chunks = splitter.split_documents([Document(t) for t in init_texts])
    ids = _doc_ids(1, len(chunks))
    collection.add(ids=ids, documents=[c.page_content for c in chunks], metadatas=[{"source": "init"} for _ in chunks])
    logger.info(f"初始化知识库：{len(chunks)} 个块")
else:
    logger.info(f"知识库已加载：{_doc_count()} 个块")

# =============================================
# DeepSeek LLM
# =============================================
llm_client = httpx.Client(timeout=30)

def call_llm(messages, tools=None):
    body = {
        "model": "deepseek-v4-flash", "messages": messages,
        "temperature": 0.3, "thinking": {"type": "disabled"}, "stream": False,
    }
    if tools:
        body["tools"] = tools
    r = llm_client.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

def _deepseek_ask(system: str, user: str) -> str:
    body = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2, "thinking": {"type": "disabled"},
    }
    r = llm_client.post("https://api.deepseek.com/chat/completions",
        json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    return r.json()["choices"][0]["message"]["content"]

# =============================================
# 工具定义 + 实现
# =============================================

TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "搜索知识库（向量语义检索），查找与问题相关的文档",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "add_document",
        "description": "向知识库添加一条新知识",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "summarize",
        "description": "对一段文本进行摘要总结",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "translate",
        "description": "将文本翻译为目标语言",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]},
    }},
]

def _tool_search(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=3)
    docs = results.get("documents", [[]])[0]
    return "\n".join(docs) if docs else "未找到相关信息"

def _tool_add(title: str, content: str) -> str:
    full = f"{title}：{content}"
    chunks = splitter.split_documents([Document(full)])
    ids = _doc_ids(_doc_count() + 1, len(chunks))
    collection.add(ids=ids, documents=[c.page_content for c in chunks], metadatas=[{"source": title} for _ in chunks])
    return f"添加成功（{len(chunks)} 个分块），共 {_doc_count()} 个块"

def _tool_summarize(text: str) -> str:
    return _deepseek_ask("You are a summarizer.", f"Summarize:\n\n{text}")

def _tool_translate(text: str, target: str) -> str:
    return _deepseek_ask(f"Translate to {target}. Output only the translation.", text)

TOOL_IMPLS = {
    "search_knowledge": _tool_search,
    "add_document": _tool_add,
    "summarize": _tool_summarize,
    "translate": _tool_translate,
}

SYSTEM_PROMPT = (
    "You are an AI assistant with dedicated tools.\n"
    "Available tools: search_knowledge, add_document, summarize, translate.\n"
    "Rules:\n"
    "1. FOR TECHNICAL QUESTIONS, use search_knowledge first.\n"
    "2. For chat, answer directly.\n"
    "3. When asked to SUMMARIZE, call summarize.\n"
    "4. When asked to TRANSLATE, call translate.\n"
    "5. Answer in the same language as the user."
)

def rag_with_fc(query: str, trace_id: str = uuid.uuid4().hex) -> str:
    _log(trace_id, "rag_start", query=query[:80])
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    tool_rounds = 0
    for _ in range(8):
        msg = call_llm(msgs, tools=TOOLS)
        if not msg.get("tool_calls"):
            _log(trace_id, "rag_done", rounds=tool_rounds)
            return msg["content"]
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
        for tc in msg["tool_calls"]:
            fname = tc["function"]["name"]
            fargs = json.loads(tc["function"]["arguments"] or "{}")
            result = TOOL_IMPLS[fname](**fargs)
            _log_tool(trace_id, fname, fargs, result)
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
            tool_rounds += 1
    _log(trace_id, "rag_max_rounds", rounds=tool_rounds)
    return msgs[-1].get("content", "")

async def stream_rag(query: str, trace_id: str):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    for _ in range(8):
        msg = call_llm(msgs, tools=TOOLS)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fname = tc["function"]["name"]
                fargs = json.loads(tc["function"]["arguments"] or "{}")
                yield f"data: {json.dumps({'type': 'tool', 'name': fname, 'args': fargs})}\n\n"
            msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
            for tc in msg["tool_calls"]:
                fname = tc["function"]["name"]
                fargs = json.loads(tc["function"]["arguments"] or "{}")
                result = TOOL_IMPLS[fname](**fargs)
                _log_tool(trace_id, fname, fargs, result)
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
            continue
        body = {
            "model": "deepseek-v4-flash", "messages": msgs,
            "temperature": 0.3, "thinking": {"type": "disabled"}, "stream": True,
        }
        async with httpx.AsyncClient(timeout=30) as ac:
            async with ac.stream("POST", "https://api.deepseek.com/chat/completions",
                json=body, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
        break
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

# =============================================
# FastAPI 应用
# =============================================

app = FastAPI(title="RAG Agent API (Production)", version="1.3.0")
app.middleware("http")(logging_middleware)
app.middleware("http")(security_middleware)

class QueryRequest(BaseModel):
    question: str

class DocRequest(BaseModel):
    title: str
    content: str

@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "chunks": _doc_count(),
        "tools": list(TOOL_IMPLS.keys()),
        "auth_required": True,
        "rate_limit": f"{RATE_LIMIT}/min",
        "version": "1.3.0",
    }

@app.post("/query")
def query(req: QueryRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    trace_id = request.headers.get(TRACE_HEADER, uuid.uuid4().hex)
    answer = rag_with_fc(req.question, trace_id)
    return {"answer": answer, "trace_id": trace_id}

@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    trace_id = request.headers.get(TRACE_HEADER, uuid.uuid4().hex)
    return StreamingResponse(
        stream_rag(req.question, trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", "Connection": "keep-alive",
            "X-Accel-Buffering": "no", "X-Trace-Id": trace_id,
        },
    )

@app.post("/doc")
def add_doc(req: DocRequest, request: Request):
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(400, "标题和内容不能为空")
    trace_id = request.headers.get(TRACE_HEADER, uuid.uuid4().hex)
    result = _tool_add(req.title, req.content)
    _log(trace_id, "doc_added", title=req.title[:40], chunks=result)
    return {"message": result, "total_chunks": _doc_count(), "trace_id": trace_id}

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 50)
    logger.info("RAG Agent API (Production v1.3.0)")
    logger.info(f"API Key 鉴权：启用")
    logger.info(f"速率限制：{RATE_LIMIT} 次/分钟")
    logger.info(f"知识库：{_doc_count()} 个块")
    logger.info(f"结构化日志：启用")
    logger.info(f"用法：X-API-Key header + X-Trace-Id header(可选)")
    logger.info("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
