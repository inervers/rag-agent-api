"""测试 Day6 API Key 鉴权"""
import sys, os
sys.path.insert(0, r"C:\Users\inervers\AppData\Roaming\Python\Python313\site-packages")

import subprocess, time, httpx, json

script = os.path.join(os.path.dirname(__file__), "rag_api.py")

proc = subprocess.Popen(
    [sys.executable, script],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=os.path.dirname(__file__),
)
time.sleep(20)

def test(desc, url, method="POST", headers=None, body=None):
    print(f"\n{desc}")
    try:
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=15) as c:
            if method == "GET":
                r = c.get(url, headers=headers)
            else:
                r = c.post(url, headers=headers, json=body)
            print(f"  Status: {r.status_code}")
            print(f"  Body: {r.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

# 1. 健康检查（免鉴权）
test("1. 健康检查（免鉴权）", "/health", "GET")

# 2. query 不带 key
test("2. query 不带 Key", "/query", body={"question": "hi"})

# 3. query 带错 key
test("3. query 带错误 Key", "/query",
     headers={"X-API-Key": "wrong-key"}, body={"question": "hi"})

# 4. query 带正确 key
test("4. query 带正确 Key", "/query",
     headers={"X-API-Key": "rag-secret-key-2024"}, body={"question": "hi"})

# 5. stream 带正确 key
test("5. stream 带正确 Key", "/query/stream",
     headers={"X-API-Key": "rag-secret-key-2024"}, body={"question": "hi"})

# 6. add doc 带正确 key
test("6. add doc 带正确 Key", "/doc",
     headers={"X-API-Key": "rag-secret-key-2024"},
     body={"title": "test", "content": "test content"})

proc.terminate()
proc.wait()
print("\n✅ 测试完成")
