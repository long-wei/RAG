import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json

def load_standardized_text(file_path: str) -> str:
    """加载标准化后的文本"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def mark_special_blocks(text: str) -> str:
    """标记特殊内容（代码块、公式）的边界，避免分块时拆分"""
    # 代码块标记
    text = re.sub(r'===代码开始===', '[[CODE_START]]', text)
    text = re.sub(r'===代码结束===', '[[CODE_END]]', text)

    # 公式标记
    text = re.sub(r'\$', '[[FORMULA_DELIM]]', text)
    return text


def get_semantic_splitter() -> RecursiveCharacterTextSplitter:
    """创建基于语义逻辑的文本分块器"""
    # 分隔符优先级：章节标题 > 小节标题 > 空行 > 句号 > 逗号
    separators = [
        '\n# ',  # 章节标题（如“# 1. 绪论”）
        '\n## ',  # 小节标题（如“## 1.2 自然语言处理的难点”）
        '\n### ',  # 子小节标题
        '\n\n',  # 空行（段落分隔）
        '\n',  # 换行
        '. ', '。',  # 中文/英文句号
        ', ', '，'  # 中文/英文逗号
    ]

    return RecursiveCharacterTextSplitter(
        chunk_size=700,  # 单块字符数
        chunk_overlap=100,  # 块间重叠字符数（保留上下文）
        separators=separators,
        length_function=len  # 按字符数计算长度
    )


def split_text_semantically(text: str, splitter: RecursiveCharacterTextSplitter) -> List[str]:
    """执行语义分块并修复特殊内容"""
    marked_text = mark_special_blocks(text)
    chunks = splitter.split_text(marked_text)

    restored_chunks = []
    for chunk in chunks:
        restored = chunk.replace('[[CODE_START]]', '===代码开始===') \
                       .replace('[[CODE_END]]', '===代码结束===') \
                       .replace('[[FORMULA_DELIM]]', '$')
        restored_chunks.append(restored)

    # 代码块合并
    fixed_chunks = []
    i = 0
    while i < len(restored_chunks):
        current = restored_chunks[i]
        start_count = current.count('===代码开始===')
        end_count = current.count('===代码结束===')
        
        # 跨块代码处理
        while i < len(restored_chunks) - 1 and start_count > end_count:
            next_idx = i + 1
            current += restored_chunks[next_idx]
            i = next_idx  # 更新索引
            start_count = current.count('===代码开始===')
            end_count = current.count('===代码结束===')
        
        fixed_chunks.append(current.strip())
        i += 1

    return fixed_chunks


def validate_chunks(chunks: List[str]) -> List[Dict]:
    """验证分块并添加元数据"""
    validated = []
    formula_pattern = re.compile(r'\$$(?:[^$]|\\\\\$)*\$$')  # 改进公式匹配模式
    
    for idx, chunk in enumerate(chunks):
        # 使用预编译正则
        code_start = chunk.count('===代码开始===')
        code_end = chunk.count('===代码结束===')
        
        # 公式验证
        formulas = formula_pattern.findall(chunk)
        formula_valid = all(len(f.split('$')) % 2 == 0 for f in formulas)  # 确保公式成对出现

        validated.append({
            "chunk_id": f"chunk_{idx}",
            "content": chunk,
            "length": len(chunk),
            "has_code": code_start > 0,
            "code_integrity": code_start == code_end,
            "formula_integrity": formula_valid
        })
    return validated





def batch_process_chunks(input_dir: str, output_dir: str):
    """批量处理目录下的所有文本文件，生成语义分块"""
    os.makedirs(output_dir, exist_ok=True)
    splitter = get_semantic_splitter()

    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(input_dir, filename)
        print(f"处理文件：{filename}")

        # 加载并分块
        text = load_standardized_text(file_path)
        chunks = split_text_semantically(text, splitter)

        # 验证并添加元数据
        validated_chunks = validate_chunks(chunks)

        # 保存分块结果（JSONL格式，每行一个块）
        output_path = os.path.join(output_dir, f"{filename}_chunks.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in validated_chunks:
                # 添加来源信息
                chunk["source"] = filename
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"生成{len(validated_chunks)}个分块，保存至{output_path}\n")

def chunking():
    # 执行批量分块
    batch_process_chunks(
        input_dir="standardized_texts",  # 输入：标准化后的文本目录
        output_dir="semantic_chunks"  # 输出：语义分块目录
    )

