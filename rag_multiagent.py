"""
rag_multiagent.py — L7 Multi-Agent 编排模块
============================================
结构化通信 + 持久化记忆 + 执行追踪。
可直接调用，也可通过 API 触发。
"""

import json, os, time, uuid
from datetime import datetime
from typing import Optional
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================
# 持久化记忆
# =============================================

class AgentMemory:
    """
    Agent 持久化记忆。每个 Agent 一个 JSON 文件。
    行动前查记忆 → 行动后存记忆。
    """

    def __init__(self, agent_name: str, base_dir: str = None):
        self.agent_name = agent_name
        memory_dir = os.path.join(base_dir or BASE_DIR, "memory")
        os.makedirs(memory_dir, exist_ok=True)
        self.filepath = os.path.join(memory_dir, f"{agent_name}.json")
        self.memories: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def add(self, entry: dict):
        entry["id"] = uuid.uuid4().hex[:8]
        entry["timestamp"] = datetime.now().isoformat()
        self.memories.append(entry)
        self._save()

    def query(self, task: str, top_k: int = 3) -> list[dict]:
        keywords = set(task.lower().split())
        scored = []
        for m in self.memories:
            text = (m.get("task", "") + " " + m.get("outcome", "") + " " +
                    " ".join(m.get("issues", []))).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:top_k]]

    def size(self) -> int:
        return len(self.memories)


# =============================================
# 执行追踪
# =============================================

class TraceLogger:
    """结构化日志：每一步都有 trace_id、Agent、耗时、状态"""

    def __init__(self):
        self.trace_id = uuid.uuid4().hex[:12]
        self.events: list[dict] = []
        self.round = 0

    def log(self, agent: str, action: str, status: str,
            detail: str = "", duration: float = 0.0, **extra):
        self.round += 1
        event = {
            "round": self.round,
            "trace_id": self.trace_id,
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "status": status,
            "duration_s": round(duration, 2),
            "detail": detail[:120],
            **extra
        }
        self.events.append(event)
        return event

    def summary(self) -> dict:
        from collections import defaultdict
        if not self.events:
            return {"error": "no events"}

        by_agent = defaultdict(list)
        for e in self.events:
            by_agent[e["agent"]].append(e)

        agent_metrics = {}
        for agent, evts in by_agent.items():
            durations = [e["duration_s"] for e in evts if e["duration_s"] > 0]
            success = [e for e in evts if e["status"] == "ok"]
            agent_metrics[agent] = {
                "calls": len(evts),
                "success": len(success),
                "avg_duration_s": round(sum(durations) / len(durations), 2) if durations else 0,
                "total_duration_s": round(sum(durations), 2),
            }

        bottleneck = max(agent_metrics.items(),
                         key=lambda x: x[1]["avg_duration_s"])

        return {
            "trace_id": self.trace_id,
            "total_events": len(self.events),
            "agent_metrics": dict(agent_metrics),
            "bottleneck": bottleneck[0],
        }


# =============================================
# Multi-Agent 工作流
# =============================================

class MultiAgentWorkflow:
    """
    Researcher → Writer → Reviewer
    带持久化记忆 + 执行追踪。

    注意：使用 OpenAI 兼容 API（DeepSeek / 其他）。
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.trace = TraceLogger()

    def _call_llm(self, system: str, user: str,
                  temperature: float = 0.3) -> str:
        start = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=1024
            )
            result = resp.choices[0].message.content
            tokens = resp.usage.total_tokens if resp.usage else 0
            self.trace.log("llm", "call", "ok",
                           detail=f"tokens={tokens}",
                           duration=time.time() - start)
            return result
        except Exception as e:
            self.trace.log("llm", "call", "fail",
                           detail=str(e)[:80],
                           duration=time.time() - start)
            raise

    def run(self, topic: str, max_retries: int = 2,
            api_key_field: str = "") -> dict:
        """
        运行完整的 Multi-Agent 写作流水线。

        返回：
            {
                "topic": "...",
                "passed": True/False,
                "rating": int,
                "attempts": int,
                "duration_s": float,
                "article": "...",
                "trace_id": "...",
                "monitor": {...},
                "memory_sizes": {...}
            }
        """
        start = time.time()

        # 初始化记忆
        researcher_mem = AgentMemory("researcher")
        writer_mem = AgentMemory("writer")
        reviewer_mem = AgentMemory("reviewer")

        # === 研究员 ===
        self.trace.log("researcher", "research", "ok", detail=f"topic={topic[:40]}")
        research = self._call_llm(
            "你是研究员。输出 JSON：{\"key_points\": [\"...\"], \"confidence\": 0-1}",
            f"研究：{topic}",
            temperature=0.1
        )
        researcher_mem.add({"task": topic, "outcome": research[:200], "role": "research"})

        # === 写作 + 审核循环 ===
        article = ""
        final_rating = 0
        passed = False
        previous_rating = 0

        for attempt in range(1, max_retries + 2):
            self.trace.log("writer", "write", "ok",
                           detail=f"round={attempt}, topic={topic[:30]}")

            # 查记忆：之前为什么被驳回
            mem_context = ""
            mems = writer_mem.query(topic)
            if mems:
                mem_context = "\n\n历史反馈：\n" + "\n".join(
                    f"- {m.get('outcome', '')[:100]}" for m in mems
                )

            article = self._call_llm(
                "你是科普写作者。输出 JSON：{\"title\": \"...\", \"content\": \"...\", \"word_count\": 0}"
                + (f"\n\n这是第 {attempt} 次修改，请改进之前的不足。" if attempt > 1 else ""),
                f"主题：{topic}\n研究资料：{research}\n{mem_context}",
                temperature=0.4
            )
            writer_mem.add({"task": f"写作{topic}第{attempt}稿",
                           "outcome": article[:200], "round": attempt})

            # 审核
            self.trace.log("reviewer", "review", "ok", detail=f"round={attempt}")
            review_raw = self._call_llm(
                "你是严格的内容审核员。输出 JSON：{\"issues\": [...], \"rating\": 1-5, \"verdict\": \"通过/需要修改\"}\n"
                "评分低于 4 必须输出需要修改。",
                f"审核文章：\n{article[:1500]}\n\n参考：{research}",
                temperature=0.1
            )

            # 解析审核结果
            cleaned = review_raw.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                review = json.loads(cleaned)
            except json.JSONDecodeError:
                review = {"issues": ["解析失败"], "rating": 3, "verdict": "需要修改"}

            final_rating = review.get("rating", 0)
            verdict = review.get("verdict", "需要修改")

            reviewer_mem.add({
                "task": f"审核{topic}第{attempt}稿",
                "outcome": f"评分{final_rating}，裁决{verdict}",
                "rating": final_rating,
                "issues": review.get("issues", [])
            })

            # 通过判定：绝对达标或相对改进
            if verdict == "通过" or final_rating >= 4:
                passed = True
                break

            if previous_rating > 0 and final_rating > previous_rating:
                passed = True
                break

            previous_rating = final_rating

            if attempt > max_retries:
                break

        elapsed = round(time.time() - start, 1)

        # 监控摘要
        monitor = self.trace.summary()

        return {
            "topic": topic,
            "passed": passed,
            "rating": final_rating,
            "verdict": verdict,
            "attempts": attempt,
            "duration_s": elapsed,
            "article": article[:500],
            "trace_id": self.trace.trace_id,
            "monitor": monitor,
            "memory_sizes": {
                "researcher": researcher_mem.size(),
                "writer": writer_mem.size(),
                "reviewer": reviewer_mem.size(),
            }
        }


# =============================================
# 快速验证
# =============================================

if __name__ == "__main__":
    import os
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("需要设置 DEEPSEEK_API_KEY")
        exit(1)

    wf = MultiAgentWorkflow(api_key=key)
    result = wf.run("多层感知机的反向传播", max_retries=1)
    print(f"\n主题: {result['topic']}")
    print(f"结果: {'✓ 通过' if result['passed'] else '✗ 未通过'}  |  "
          f"评分: {result['rating']}/5  |  尝试: {result['attempts']} 次")
    print(f"耗时: {result['duration_s']}s")
    print(f"追踪 ID: {result['trace_id']}")
    print(f"监控: {result['monitor']}")
