from transformers import AutoTokenizer, AutoModel
import torch
import re
import faiss
import os
import json
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from tqdm import tqdm

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_tech_model(model_name_or_path: str = "models") -> tuple:
    """加载技术文档的预训练模型（使用Hugging Face API）"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)   # paraphrase-multilingual-mpnet-base-v2模型
    model = AutoModel.from_pretrained(model_name_or_path)
    print(f"模型加载完成，向量维度：{model.config.hidden_size}")
    return tokenizer, model


def load_chunks(chunk_dir: str) -> List[Dict]:
    """加载所有语义分块数据"""
    chunks = []
    for filename in os.listdir(chunk_dir):
        if not filename.endswith("_chunks.jsonl"):
            continue
        with open(os.path.join(chunk_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    print(f"共加载{len(chunks)}个分块")
    return chunks

def preprocess_tech_content(content: str) -> str:
    """预处理技术内容，强化特殊结构的语义权重"""
    # 代码块标记
    content = re.sub(r'===代码开始===', '代码块：\n===代码开始===', content)
    # 公式标记
    content = re.sub(r'\$([^$]+)\$', r'公式：\$\1\$', content)
    # 去除过长空格，保留结构的同时压缩冗余
    content = re.sub(r' +', ' ', content)
    return content


def generate_embeddings(chunks: List[Dict], tokenizer, model) -> List[Dict]:
    """生成分块向量并添加到数据中（使用Hugging Face API）"""
    contents = [preprocess_tech_content(chunk["content"]) for chunk in chunks]

    # 添加torch.no_grad()减少内存占用
    with torch.no_grad():
        # 批量生成向量（优化批次大小）
        if model.device.type == "cuda":
            # 获取GPU内存信息并动态调整批次大小
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory > 10 * 1024**3:  # 10GB以上
                batch_size = 64
            elif total_memory > 6 * 1024**3:  # 6GB以上
                batch_size = 32
            elif total_memory > 4 * 1024**3:  # 4GB以上
                batch_size = 16
            else:
                batch_size = 8
            print(f"使用GPU内存: {total_memory / (1024**3):.2f}GB, 批次大小: {batch_size}")
        else:
            batch_size = 8
            print(f"使用CPU, 批次大小: {batch_size}")

        embeddings = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            
            # Tokenize sentences
            encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(model.device)
            
            # Compute token embeddings
            model_output = model(**encoded_input)
            
            # Perform pooling
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Convert to numpy array and add to embeddings list
            embeddings.extend(batch_embeddings.cpu().numpy())

    # 向量添加到分块数据
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()  # 转为列表便于存储
    return chunks


def validate_embeddings(chunks: List[Dict], sample_size: int = 50) -> float:
    """验证向量质量：计算同章节分块的平均余弦相似度"""
    # 按章节聚合
    chapter_chunks = {}
    for chunk in chunks:
        chapter = chunk["source"].split("_")[0]
        if chapter not in chapter_chunks:
            chapter_chunks[chapter] = []
        chapter_chunks[chapter].append(np.array(chunk["embedding"]))

    # 计算所有同章节组合的相似度
    similarities = []
    for chapter, embeddings in chapter_chunks.items():
        if len(embeddings) < 2:
            continue
        # 计算所有组合的余弦相似度
        for emb1, emb2 in combinations(embeddings, 2):
            similarities.append(cosine_similarity(emb1.reshape(1,-1), emb2.reshape(1,-1))[0][0])

    avg_sim = np.mean(similarities) if similarities else 0.0
    print(f"同章节分块平均余弦相似度：{avg_sim:.4f}（≥0.6为合格）")
    return avg_sim


def build_faiss_index(chunks: List[Dict], output_path: str):
    """构建FAISS向量索引并保存"""
    embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
    dimension = embeddings.shape[1]
    
    # Flat索引、精确搜索
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS索引构建完成，向量数量：{index.ntotal}")
    
    # 元数据路径处理
    metadata_path = os.path.splitext(output_path)[0] + "_metadata.jsonl"
    faiss.write_index(index, output_path)
    # 保存元数据进度条
    with open(metadata_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="保存元数据"):
            meta = {k: v for k, v in chunk.items() if k in ["chunk_id", "source", "content"]}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return index


def test_retrieval(query: str, tokenizer, model, index, metadata_path: str, top_k: int = 3):
    """测试检索效果：查询技术问题并返回最相关分块"""
    #生成查询向量
    query = preprocess_tech_content(query)
    encoded_input = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        query_emb = mean_pooling(model_output, encoded_input['attention_mask'])
    query_emb = query_emb.cpu().numpy().astype(np.float32)

    # 检索Top-K结果
    distances, indices = index.search(query_emb, top_k)

    # 显式指定UTF-8编码读取元数据文件
    with open(metadata_path, "r", encoding="utf-8") as f:  # 修复：添加encoding参数
        metadata = [json.loads(line) for line in f]
    print(f"\n查询：{query}")
    for i, idx in enumerate(indices[0]):
        print(f"\nTop-{i + 1}（距离：{distances[0][i]:.4f}）：")
        print(f"来源：{metadata[idx]['source']}")
        print(f"内容：{metadata[idx]['content'][:200]}...")  # 显示前200个字符

def vectorization():
    try:
        tokenizer, model = load_tech_model("../models")
        chunks = load_chunks("semantic_chunks")
        chunks_with_emb = generate_embeddings(chunks, tokenizer, model)
        validate_embeddings(chunks_with_emb)
        
        # 路径处理
        index_path = os.path.join("../tech_doc_index.index")
        metadata_path = os.path.splitext(index_path)[0] + "_metadata.jsonl"
        
        index = build_faiss_index(chunks_with_emb, index_path)
        test_retrieval(
            "Word2vec模型的CBOW和Skip-gram有什么区别？",
            tokenizer, model,
            index,
            metadata_path
        )
    except Exception as e:
        print(f"处理过程中出错: {e}")
        raise
