import numpy as np
import faiss
from typing import List, Dict, Tuple
import math


def dynamic_top_k(query_length: int) -> int:
    """
    根据查询长度动态调整检索数量（top_k）：
    - 短查询（<20字）：聚焦核心信息，取top_k=3
    - 中长查询（20-50字）：平衡广度与精度，取top_k=5
    - 长查询（>50字）：覆盖更多相关内容，取top_k=7
    """
    if query_length < 20:
        return 3
    elif 20 <= query_length <= 50:
        return 6
    else:
        return 9


def filter_low_relevance(chunks: List[Dict], distance_threshold: float = 1.2) -> List[Dict]:
    """
    过滤低相关性分块：基于L2距离（越小越相关），过滤距离超过阈值的结果
    阈值根据技术文档分块向量分布统计得出（实验验证1.2为最优阈值）
    """
    return [chunk for chunk in chunks if chunk["distance"] <= distance_threshold]


def deduplicate_by_source(chunks: List[Dict]) -> List[Dict]:
    """
    按来源去重：同一文档（source）保留距离最小的分块，避免重复信息
    """
    source_map = {}
    for chunk in chunks:
        source = chunk["source"]
        if source not in source_map or chunk["distance"] < source_map[source]["distance"]:
            source_map[source] = chunk
    return sorted(source_map.values(), key=lambda x: x["distance"])  # 按距离升序排序


def retrieve_and_optimize(
        query_embedding: np.ndarray,
        index: faiss.Index,
        metadata: List[Dict],
        query_text: str,
        distance_threshold: float = 9.0
) -> List[Dict]:
    """
    检索并优化相关分块的完整流程：
    1. 动态确定top_k
    2. 基础检索
    3. 过滤低相关性结果
    # 4. 按来源去重
    5. 最终截断至合理数量（最多8个）
    """
    # 1. 动态计算top_k
    top_k = dynamic_top_k(len(query_text))

    # 2. FAISS基础检索（返回距离和索引）
    distances, indices = index.search(query_embedding, top_k)

    # 3. 组装原始检索结果（包含距离、来源、内容）
    raw_chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = metadata[idx].copy()
        chunk["distance"] = float(dist)  # 转为float便于JSON序列化
        raw_chunks.append(chunk)

    # 4. 优化：过滤低相关性
    filtered_chunks = filter_low_relevance(raw_chunks, distance_threshold)
    # 查看原始检索结果
    print(f"原始检索结果数量: {len(raw_chunks)}")
    for i, chunk in enumerate(raw_chunks):
        print(f"原始结果{i + 1}: 距离={chunk['distance']:.4f}")

    # 查看过滤后的结果
    print(f"过滤后结果数量: {len(filtered_chunks)}")

    if not filtered_chunks:
        return []  # 无相关结果

    # 5. 优化：去重
    deduped_chunks = deduplicate_by_source(filtered_chunks)

    # 6. 最终截断（最多保留8个，避免上下文过长）
    return deduped_chunks[:8]
    # return filtered_chunks[:8]
    # return raw_chunks


# 检索质量评估辅助函数
def calculate_retrieval_precision(retrieved_chunks: List[Dict], relevant_sources: List[str]) -> float:
    """
    评估检索精度：检索结果中相关来源的占比
    relevant_sources：人工标注的与查询相关的文档来源列表
    """
    if not retrieved_chunks:
        return 0.0
    relevant_count = sum(1 for chunk in retrieved_chunks if chunk["source"] in relevant_sources)
    return relevant_count / len(retrieved_chunks)


# 测试代码
if __name__ == "__main__":
    # 加载依赖组件（需先运行索引与模型加载模块）
    from index_loader import load_all_components
    from query_processor import generate_query_embedding

    # 加载模型、索引、元数据
    tokenizer, model, index, metadata = load_all_components()

    # 测试查询
    # test_query = "Word2vec模型的CBOW和Skip-gram有什么区别？"
    # test_query = "什么是自然语言处理"
    test_query = ["Transform","Word2vec"]
    print(f"测试查询：{test_query}")

    # 生成查询向量
    query_emb = generate_query_embedding(test_query, tokenizer, model)

    # 检索并优化
    optimized_chunks = retrieve_and_optimize(
        query_embedding=query_emb,
        index=index,
        metadata=metadata,
        query_text=test_query
    )

    # 输出结果
    print(f"\n优化后检索结果（{len(optimized_chunks)}个）：")
    for i, chunk in enumerate(optimized_chunks, 1):
        print(f"\nTop-{i}（距离：{chunk['distance']:.4f} | 来源：{chunk['source']}）")
        print(f"内容片段：{chunk['content'][:300]}...")  # 显示前300字符

