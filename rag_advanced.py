"""
rag_advanced.py — L6 进阶检索模块
=================================
多路召回（稠密 + BM25）+ Cross-Encoder Reranker
直接复用 rag_api.py 中的 Chroma 集合和嵌入模型。
"""

import re, math, time, json, os
from typing import Optional

# rank_bm25：pip install rank_bm25
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# sentence-transformers：用于 Cross-Encoder
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class HybridSearch:
    """
    多路召回：稠密检索（Chroma）+ 稀疏检索（BM25）+ RRF 融合。
    直接复用外部传入的 Chroma collection 和 embed 函数。
    """

    def __init__(self, collection, embed_func, corpus_docs: list[str] = None):
        self.collection = collection
        self.embed_func = embed_func
        self.bm25 = None
        self.corpus_docs = corpus_docs or []
        if BM25Okapi is not None and self.corpus_docs:
            tokenized = [re.findall(r"\w+", d.lower()) for d in self.corpus_docs]
            self.bm25 = BM25Okapi(tokenized)

    def set_corpus(self, docs: list[str]):
        """设置/更新 BM25 语料库"""
        self.corpus_docs = docs
        if BM25Okapi is not None:
            tokenized = [re.findall(r"\w+", d.lower()) for d in docs]
            self.bm25 = BM25Okapi(tokenized)

    def dense_search(self, query: str, top_k: int = 10) -> list[dict]:
        """稠密检索"""
        results = self.collection.query(
            query_texts=[query], n_results=top_k
        )
        items = []
        for doc_id, doc, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        ):
            items.append({
                "id": doc_id,
                "text": doc,
                "score": round(1 - dist, 4),
            })
        return items

    def sparse_search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 稀疏检索"""
        if self.bm25 is None:
            return []
        tokens = re.findall(r"\w+", query.lower())
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            {"id": f"doc_{idx}", "text": self.corpus_docs[idx], "score": round(score, 4)}
            for idx, score in ranked
            if score > 0
        ]

    @staticmethod
    def rrf_fusion(dense: list[dict], sparse: list[dict],
                   k: int = 60, weights: tuple = (1.0, 1.0)) -> list[dict]:
        """
        Reciprocal Rank Fusion。

        参数：
            k: 排名衰减常数（越大 fusion 越平滑）
            weights: (稠密权重, 稀疏权重) — 默认均等
        """
        scores = {}
        doc_map = {}

        for rank, item in enumerate(dense):
            scores[item["id"]] = scores.get(item["id"], 0) + weights[0] / (k + rank + 1)
            doc_map[item["id"]] = item["text"]

        for rank, item in enumerate(sparse):
            scores[item["id"]] = scores.get(item["id"], 0) + weights[1] / (k + rank + 1)
            doc_map[item["id"]] = item["text"]

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"id": doc_id, "text": doc_map[doc_id], "rrf_score": round(score, 4)}
            for doc_id, score in fused
        ]

    def search(self, query: str, top_k: int = 10,
               dense_weight: float = 1.0, sparse_weight: float = 1.0) -> dict:
        """多路召回入口：稠密 + 稀疏 + RRF"""

        dense = self.dense_search(query, top_k)
        sparse = self.sparse_search(query, top_k)

        # 稠密单独结果
        dense_only = dense[:5] if dense else []

        # 混合
        hybrid = self.rrf_fusion(dense, sparse, weights=(dense_weight, sparse_weight))[:top_k]

        # 统计：稠密和稀疏的重叠比例
        dense_ids = set(item["id"] for item in dense)
        sparse_ids = set(item["id"] for item in sparse)
        overlap = len(dense_ids & sparse_ids)

        return {
            "query": query,
            "dense_top": dense_only,
            "hybrid_top": hybrid,
            "stats": {
                "dense_count": len(dense),
                "sparse_count": len(sparse),
                "overlap": overlap,
            }
        }


class Reranker:
    """
    Cross-Encoder 重排序。
    对候选文档做逐对精确打分，重新排序。
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if CrossEncoder is not None:
            try:
                path = model_path or os.path.join(
                    os.path.dirname(__file__), "models", "cross-encoder"
                )
                self.model = CrossEncoder(path)
            except Exception as e:
                print(f"Reranker 加载失败（可回退）: {e}")

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = 5) -> list[dict]:
        """
        对候选项做 Cross-Encoder 重排。

        candidates: [{"id": "...", "text": "...", "score": 0.xx}, ...]
        返回：[{"id": "...", "text": "...", "bi_score": 0.xx, "ce_score": 0.xx}, ...]
        """
        if self.model is None or not candidates:
            return [{"id": c["id"], "text": c["text"],
                     "bi_score": c.get("score", 0), "ce_score": c.get("score", 0)}
                    for c in candidates]

        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.model.predict(pairs)

        reranked = sorted(
            zip(candidates, ce_scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            {"id": c["id"], "text": c["text"],
             "bi_score": c.get("score", 0),
             "ce_score": round(float(score), 4)}
            for c, score in reranked
        ]


# =============================================
# 快速验证（独立运行时）
# =============================================

if __name__ == "__main__":
    print("=== HybridSearch 快速测试 ===")
    from rag_api import collection, embed_texts

    # 获取知识库中的文档作为 BM25 语料
    all_docs = collection.get()
    corpus = all_docs.get("documents", [])

    hs = HybridSearch(collection, embed_texts, corpus)

    result = hs.search("What is PyTorch?")
    print(f"\n查询: {result['query']}")
    print(f"\n稠密 Top:")
    for item in result["dense_top"]:
        print(f"  [{item['id']}] score={item['score']}  {item['text'][:60]}...")
    print(f"\n混合 Top:")
    for item in result["hybrid_top"]:
        print(f"  [{item['id']}] rrf={item['rrf_score']}  {item['text'][:60]}...")

    print(f"\n统计: {result['stats']}")
