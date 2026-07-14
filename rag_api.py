"""L5-Day1: FC RAG API + API Key 鉴权

生产化第一步：所有接口加 API Key 校验。
客户端需要在请求头中带 X-API-Key，否则返回 401。
"""

import sys, os, json

_REAL_USER_SITE = r"C:\Users\inervers\AppData\Roaming\Python\Python313\site-packages"
if os.path.isdir(_REAL_USER_SITE) and _REAL_USER_SITE not in sys.path:
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

# 从环境变量读取（.env 已注入）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
RAG_API_KEY = os.environ.get("RAG_API_KEY", "rag-secret-key-2024")  # 默认 key，生产环境请修改

if not DEEPSEEK_API_KEY:
    print("需要设置 DEEPSEEK_API_KEY")
    exit(1)

import time as time_module
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
# 速率限制器
# =============================================

RATE_LIMIT = int(os.environ.get("RAG_RATE_LIMIT", "10"))  # 每分钟最多请求数
WINDOW_SEC = 60

class RateLimiter:
    """滑动窗口速率限制器。记录每个 Key 的请求时间戳，超窗口的自动淘汰"""
    def __init__(self):
        self._records: dict[str, list[float]] = {}

    def check(self, key: str) -> tuple[bool, int, int]:
        """
        返回：(是否允许, 当前窗口内已用次数, 限制次数)
        """
        now = time_module.time()
        window_start = now - WINDOW_SEC

        if key not in self._records:
            self._records[key] = []

        # 剔除窗口外的时间戳
        self._records[key] = [t for t in self._records[key] if t > window_start]

        used = len(self._records[key])
        if used >= RATE_LIMIT:
            return False, used, RATE_LIMIT

        self._records[key].append(now)
        return True, used + 1, RATE_LIMIT

rate_limiter = RateLimiter()

# =============================================
# 鉴权 + 限流中间件
# =============================================

AUTH_HEADER = "X-API-Key"

async def security_middleware(request: Request, call_next):
    """统一安全检查：鉴权 → 限流"""
    # /health 允许不带 key 也不限流
    if request.url.path == "/health":
        return await call_next(request)

    # 1. 鉴权
    api_key = request.headers.get(AUTH_HEADER)
    if not api_key:
        return JSONResponse(status_code=401, content={"error": "Missing X-API-Key header"})
    if api_key != RAG_API_KEY:
        return JSONResponse(status_code=403, content={"error": "Invalid API Key"})

    # 2. 限流
    allowed, used, limit = rate_limiter.check(api_key)
    if not allowed:
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
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
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

# 初始化知识库
if _doc_count() == 0:
    init_texts = [
        "Python was created by Guido van Rossum and first released in 1991. It is a high-level general-purpose programming language emphasizing code readability with significant indentation. Python supports multiple programming paradigms including structured, object-oriented, and functional programming.",
        "PyTorch was developed by Meta AI (Facebook AI Research) and released in 2016. It is an open-source machine learning framework. Key features include dynamic computation graphs, GPU-accelerated tensor computation, automatic differentiation with Autograd.",
        "The Transformer architecture was introduced by Google in the 2017 paper 'Attention Is All You Need'. It relies entirely on self-attention mechanisms to process sequential data. It is the foundation for BERT, GPT, T5, and ViT.",
        "RAG (Retrieval-Augmented Generation) combines a retriever and a generator. The retriever searches a knowledge base for relevant documents. These retrieved documents are fed as context to the LLM to produce informed answers grounded in real sources.",
        "Chroma is an open-source vector database built for AI applications. It supports cosine similarity, L2 distance, and inner product. Chroma supports persistent storage and integrates natively with LangChain and LlamaIndex.",
        "LangChain is an open-source framework for LLM application development. It provides modular abstractions for models, prompts, chains, memory, agents, and retrieval. Supports LCEL for composing pipelines.",
    ]
    chunks = splitter.split_documents([Document(t) for t in init_texts])
    ids = _doc_ids(1, len(chunks))
    collection.add(ids=ids, documents=[c.page_content for c in chunks], metadatas=[{"source": "init"} for _ in chunks])
    print(f"▶ 知识库初始化：{len(chunks)} 个块")
else:
    print(f"▶ 知识库已加载：{_doc_count()} 个块")

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
        "description": "向知识库添加一条新知识（Chroma 持久化，重启不丢）",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
            "required": ["title", "content"],
        },
    }},
    {"type": "function", "function": {
        "name": "summarize",
        "description": "对一段文本进行摘要总结",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "translate",
        "description": "将文本翻译为目标语言",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}},
            "required": ["text", "target_language"],
        },
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

def rag_with_fc(query: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    for _ in range(8):
        msg = call_llm(msgs, tools=TOOLS)
        if not msg.get("tool_calls"):
            return msg["content"]
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
        for tc in msg["tool_calls"]:
            fname = tc["function"]["name"]
            fargs = json.loads(tc["function"]["arguments"] or "{}")
            result = TOOL_IMPLS[fname](**fargs)
            print(f"  🛠  {fname}({json.dumps(fargs, ensure_ascii=False)})")
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    return msgs[-1].get("content", "")

async def stream_rag(query: str):
    """流式 FC 生成器。工具调用轮非流式，最后一轮流式"""
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
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
            continue
        # 流式返回
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

app = FastAPI(title="RAG Agent API (Production)", version="1.2.0")
app.middleware("http")(security_middleware)

class QueryRequest(BaseModel):
    question: str

class DocRequest(BaseModel):
    title: str
    content: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": _doc_count(),
        "tools": list(TOOL_IMPLS.keys()),
        "auth_required": True,
        "rate_limit": f"{RATE_LIMIT}/min",
    }

@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    return {"answer": rag_with_fc(req.question)}

@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    return StreamingResponse(
        stream_rag(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

@app.post("/doc")
def add_doc(req: DocRequest):
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(400, "标题和内容不能为空")
    result = _tool_add(req.title, req.content)
    return {"message": result, "total_chunks": _doc_count()}

if __name__ == "__main__":
    import uvicorn
    print(f"启动 RAG Agent API（Production v1.2.0）...")
    print(f"  API Key 鉴权：启用")
    print(f"  速率限制：{RATE_LIMIT} 次/分钟")
    print(f"  知识库：{_doc_count()} 个块")
    print(f"  POST /query         →  问答（需 X-API-Key 头）")
    print(f"  POST /query/stream  →  流式问答（需 X-API-Key 头）")
    print(f"  POST /doc           →  添加知识（需 X-API-Key 头）")
    print(f"  GET  /health        →  健康检查（无需鉴权）")
    print(f"\n测试方法：")
    print(f'  curl -H "X-API-Key: {RAG_API_KEY}" ...')
    uvicorn.run(app, host="0.0.0.0", port=8000)
