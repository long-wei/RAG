from transformers import AutoTokenizer, AutoModel
import faiss
import os
import json
import torch
from typing import List, Dict, Tuple


def load_tech_model(model_name_or_path: str = "paraphrase-multilingual-mpnet-base-v2") -> Tuple[AutoTokenizer, AutoModel]:
    """
    加载技术文档的预训练向量模型和Tokenizer

    Args:
        model_name_or_path: 模型路径或Hugging Face模型名

    Returns:
        tokenizer: 预训练Tokenizer
        model: 预训练向量模型（自动分配到可用设备）
    """
    # 校验模型路径
    if not os.path.exists(model_name_or_path) and not model_name_or_path in ["paraphrase-multilingual-mpnet-base-v2"]:
        raise FileNotFoundError(f"模型路径不存在: {model_name_or_path}")

    # 加载Tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    # 自动分配设备（优先GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 设置为评估模式

    print(f"模型加载完成 | 设备: {device} | 向量维度: {model.config.hidden_size}")
    return tokenizer, model


def load_faiss_index_and_metadata(
        index_path: str = "tech_doc_index.index",
        metadata_path: str = None
) -> Tuple[faiss.Index, List[Dict]]:
    """
    加载FAISS向量索引和对应的元数据

    Args:
        index_path: FAISS索引文件路径
        metadata_path: 元数据文件路径（默认自动推导为index_path同级的_metadata.jsonl）

    Returns:
        index: FAISS向量索引对象
        metadata: 分块元数据列表（包含chunk_id、source、content）
    """
    # 自动推导元数据路径
    if metadata_path is None:
        metadata_path = os.path.splitext(index_path)[0] + "_metadata.jsonl"

    # 校验文件存在性
    for path in [index_path, metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

    # 加载FAISS索引
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        raise RuntimeError(f"加载FAISS索引失败: {str(e)}")

    # 加载元数据（带进度提示）
    metadata = []
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line.strip()))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"元数据文件格式错误: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"加载元数据失败: {str(e)}")

    # 校验索引与元数据数量一致性
    if len(metadata) != index.ntotal:
        raise ValueError(
            f"索引与元数据数量不匹配 | "
            f"索引向量数: {index.ntotal} | 元数据条数: {len(metadata)}"
        )

    print(
        f"索引与元数据加载完成 | "
        f"向量数量: {index.ntotal} | "
        f"元数据条数: {len(metadata)} | "
        f"索引维度: {index.d}"
    )
    return index, metadata


def load_all_components(
        model_path: str = "paraphrase-multilingual-mpnet-base-v2",
        index_path: str = "tech_doc_index.index"
) -> Tuple[AutoTokenizer, AutoModel, faiss.Index, List[Dict]]:
    """
    一站式加载所有组件（模型、Tokenizer、索引、元数据）

    Returns:
        tokenizer: 预训练Tokenizer
        model: 预训练向量模型
        index: FAISS向量索引
        metadata: 分块元数据列表
    """
    tokenizer, model = load_tech_model(model_path)
    index, metadata = load_faiss_index_and_metadata(index_path)
    return tokenizer, model, index, metadata


# 测试代码
if __name__ == "__main__":
    try:
        # 加载所有组件
        tokenizer, model, index, metadata = load_all_components()

        # 打印关键信息
        print(f"\nTokenizer类型: {type(tokenizer)}")
        print(f"模型设备: {model.device}")
        print(f"FAISS索引类型: {type(index)}")
        print(f"元数据示例: {metadata[0] if metadata else '无'}")
    except Exception as e:
        print(f"加载失败: {str(e)}")
