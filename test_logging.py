import sys, subprocess, time, httpx

sys.path.insert(0, r"C:\Users\inervers\AppData\Roaming\Python\Python313\site-packages")

script = r"C:\Users\inervers\Desktop\OH-WorkSpace\rag-agent-api\rag_api.py"
proc = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(18)

headers = {"X-API-Key": "rag-secret-key-2024"}
client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=15)

# 健康检查
r = client.get("/health")
print(f"health: {r.status_code}")

# 问答
r = client.post("/query", headers=headers, json={"question": "What is Python?"})
data = r.json()
print(f"query: {r.status_code}, trace={data['trace_id'][:8]}")

# 自定义 trace_id
r = client.post("/query", headers={**headers, "X-Trace-Id": "my-custom-trace-001"}, json={"question": "hi"})
print(f"query+id: {r.status_code}, trace={r.json()['trace_id']}")

# 不带 key
r = client.post("/query", json={"question": "hi"})
print(f"no-key: {r.status_code}")

proc.terminate()
proc.wait()
client.close()

print()
print("=== 服务端日志 ===")
for line in proc.stderr.read().decode().splitlines():
    if "INFO" in line or "WARN" in line or "ERROR" in line:
        print(line)
