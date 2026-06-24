import re
import numpy as np
import torch
from typing import List, Union
from transformers import AutoTokenizer, AutoModel


def preprocess_tech_query(query: str) -> str:
    """
    预处理技术查询文本，强化公式、代码块和专业术语的语义特征，与分块预处理逻辑保持一致

    Args:
        query: 用户输入的原始查询（如"解释Word2vec的CBOW模型公式"）

    Returns:
        processed_query: 预处理后的查询文本
    """
    # 1. 统一特殊符号格式（与分块处理对齐）
    query = query.replace("？", "?").replace("，", ",").replace("。", ".")  # 中英文标点统一

    # 2. 强化公式标记（若查询含公式，添加"公式："前缀，引导模型关注）
    # 匹配$包裹的公式或明显数学表达式（如"y=Wx+b"）
    formula_pattern = r'(\$.*?\$)|([a-zA-Z0-9]+\s*=\s*[^=]+)'
    query = re.sub(
        formula_pattern,
        lambda m: f"公式：{m.group(0)}",  # 对匹配到的公式添加标记
        query,
        flags=re.DOTALL
    )

    # 3. 强化代码块标记（若查询含代码片段，添加"代码块："前缀）
    code_pattern = r'(`.*?`)|(def |function |import |print\()'      # 匹配代码片段或关键词
    query = re.sub(
        code_pattern,
        lambda m: f"代码块：{m.group(0)}",      # 对匹配到的代码添加标记
    query,
    flags = re.IGNORECASE
    )

    # 4. 专业术语标准化（与分块术语统一，如"词嵌入"→"词嵌入（Embedding）"）
    term_mappings = {
        r'词向量': '词向量（Word Vector）',
        r'嵌入': '嵌入（Embedding）',
        r'注意力机制': '注意力机制（Attention Mechanism）',
        r'CNN': '卷积神经网络（CNN）',
        r'RNN': '循环神经网络（RNN）'
    }
    for pattern, replacement in term_mappings.items():
        query = re.sub(pattern, replacement, query)

    # 5. 清理冗余空格和符号
    query = re.sub(r'\s+', ' ', query).strip()   # 合并多余空格
    query = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\,\?\:\$\(\)\_\^\+\-\/\=]', '', query)  # 保留关键符号

    return query


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    对模型输出进行平均池化，生成句子向量（与分块向量生成逻辑完全一致）

    Args:
        model_output: 模型最后一层的输出 (batch_size, seq_len, hidden_size)
        attention_mask: 注意力掩码，标记有效token (batch_size, seq_len)

    Returns:
        sentence_embedding: 池化后的句子向量 (batch_size, hidden_size)
    """
    token_embeddings = model_output[0]  # 取最后一层输出
    # 正确地扩展attention_mask以匹配token_embeddings的维度
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)



def generate_query_embedding(
        query: Union[str, List[str]],
        tokenizer: AutoTokenizer,
        model: AutoModel,
        batch_size: int = 8
) -> np.ndarray:
    """
    生成查询的向量表示，支持单条或批量查询，与分块向量生成逻辑严格对齐

    Args:
        query: 单条查询字符串或查询列表
        tokenizer: 预训练Tokenizer（与分块向量模型配套）
        model: 预训练向量模型（与分块向量模型配套）
        batch_size: 批量处理大小（根据设备自动调整）

    Returns:
        query_embeddings: 查询向量数组，形状为 (n_queries, hidden_size)
    """
    # 确保输入为列表格式
    if isinstance(query, str):
        queries = [query]
    else:
        queries = query

    # 预处理所有查询
    processed_queries = [preprocess_tech_query(q) for q in queries]

    # 自动调整批量大小（GPU可加大，CPU减小）
    device = model.device
    if device.type == "cuda":
        batch_size = min(batch_size, 32)  # GPU最大批量32
    else:
        batch_size = min(batch_size, 4)  # CPU最大批量4

    # 批量生成向量
    embeddings = []
    model.eval()  # 确保模型在评估模式
    with torch.inference_mode():  # 比no_grad()更高效的推理模式
        for i in range(0, len(processed_queries), batch_size):
            batch = processed_queries[i:i + batch_size]

            # Tokenize批量查询
            encoded_input = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,  # 与分块处理的max_length保持一致
                return_tensors="pt"
            ).to(device)

            # 模型前向传播
            model_output = model(**encoded_input)

            # 池化生成句子向量
            batch_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

            # 转为numpy并缓存
            embeddings.append(batch_embeddings.cpu().numpy())

    # 拼接所有批次向量
    query_embeddings = np.vstack(embeddings).astype(np.float32)  # 转为FAISS兼容的float32
    return query_embeddings


# 测试代码
if __name__ == "__main__":
    # 加载模型（需先运行索引与模型加载模块）
    from index_loader import load_tech_model  # 假设索引与模型加载模块已实现

    tokenizer, model = load_tech_model("paraphrase-multilingual-mpnet-base-v2")

    # 测试单条查询
    single_query = "Word2vec的CBOW和Skip-gram模型有什么区别？请用公式说明"
    single_emb = generate_query_embedding(single_query, tokenizer, model)
    print(f"单条查询向量形状: {single_emb.shape}")  # 应输出 (1, 768) 或模型对应维度

    # 测试批量查询
    batch_queries = [
        "解释Transformer中的自注意力机制公式$Attention(Q,K,V)$",
        "写一段用PyTorch实现词向量训练的代码"
    ]
    batch_embs = generate_query_embedding(batch_queries, tokenizer, model)
    print(f"批量查询向量形状: {batch_embs.shape}")  # 应输出 (2, 768) 或模型对应维度

    # 验证向量规范性
    assert batch_embs.dtype == np.float32, "向量类型必须为float32"
    assert not np.isnan(batch_embs).any(), "向量中不能包含NaN值"
    print("向量生成验证通过")
