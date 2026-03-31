import re
import json
import os
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import traceback
import multiprocessing

# === 预编译的正则表达式 ===
# 清理特殊字符的模式
CLEAN_SPECIAL_CHARS_PATTERN = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\,\?\!\;\:\=\(\)\{\}\[\]\<\>\$\_\^\+\-\/\|]')
# 空格清理模式
CLEAN_SPACES_PATTERN = re.compile(r' +')
CLEAN_LEADING_SPACES_PATTERN = re.compile(r'^ +', re.MULTILINE)
CLEAN_TRAILING_SPACES_PATTERN = re.compile(r' +$', re.MULTILINE)
CLEAN_EMPTY_LINES_PATTERN = re.compile(r'\n+')

# 冗余内容模式
REDUNDANT_PATTERNS = [
    re.compile(r'目录.*?正文开始', re.DOTALL | re.IGNORECASE),
    re.compile(r'小结.*?（下一章）', re.DOTALL | re.IGNORECASE),
    re.compile(r'参考文献.*?\n', re.DOTALL | re.IGNORECASE),
    re.compile(r'第\d+页')
]
FORMULA_NUMBER_PATTERN = re.compile(r'（式\d+-\d+）|(\(\d+\.\d+\))')
CODE_PROMPT_PATTERN = re.compile(r'>>>')

# 术语映射和公式处理
TERM_MAPPINGS = {
    '词向量': '词向量（Word Vector）',
    '嵌入': '嵌入（Embedding）',
    '注意力机制': '注意力机制（Attention Mechanism）',
    '循环神经网络': '循环神经网络（RNN, Recurrent Neural Network）'
}
FORMULA_FORMAT_PATTERN = re.compile(r'(?<!\$)([a-zA-Z0-9\_\^\+\-\/\(\)]+\=[^$]+)(?!\$)')
CODE_START_PATTERN = re.compile(r'===代码开始===\s+===代码开始===')
CODE_END_PATTERN = re.compile(r'===代码结束===\s+===代码结束===')
SECTION_PATTERN = re.compile(r'^(\d+\.\d+)\s+', re.MULTILINE)
CHAPTER_PATTERN = re.compile(r'^(\d+)\s+', re.MULTILINE)


def initial_clean(text: str) -> str:
    """初步清洗：去除冗余符号、空格和空行"""
    # 去除特殊符号（保留技术文档必要符号：公式$、代码=:、标点.,等）
    text = CLEAN_SPECIAL_CHARS_PATTERN.sub('', text)
    
    # 清理空格：合并连续空格，去除行首行尾空格
    text = CLEAN_SPACES_PATTERN.sub(' ', text)
    text = CLEAN_LEADING_SPACES_PATTERN.sub('', text)
    text = CLEAN_TRAILING_SPACES_PATTERN.sub('', text)
    
    # 清理空行：合并连续空行为一个
    text = CLEAN_EMPTY_LINES_PATTERN.sub('\n', text).strip()
    
    return text


def remove_redundant_content(text: str) -> str:
    """去除技术文档中的冗余内容"""
    # 去除重复的通用模块（如目录、小结）
    for pattern in REDUNDANT_PATTERNS:
        text = pattern.sub('', text)
    
    # 去除公式编号（如"(1.1)"“式3-2”）
    text = FORMULA_NUMBER_PATTERN.sub('', text)
    
    # 去除代码块标记外的多余标记（如原PDF中的“>>>”）
    text = CODE_PROMPT_PATTERN.sub('', text)
    
    return text.strip()


def standardize_terms_and_format(text: str) -> str:
    """标准化术语和格式（优化了术语替换逻辑）"""
    # 中英文术语统一 - 单次遍历完成所有替换
    def replace_term(match):
        term = match.group(0)
        return TERM_MAPPINGS.get(term, term)
    
    # 使用单个正则表达式匹配所有可能的术语
    all_terms_pattern = re.compile('|'.join(re.escape(term) for term in TERM_MAPPINGS.keys()))
    text = all_terms_pattern.sub(replace_term, text)
    
    # 规范公式格式（确保$包裹完整）
    text = FORMULA_FORMAT_PATTERN.sub(r'$\1$', text)
    
    # 规范代码块标记（确保开始和结束标记成对）
    text = CODE_START_PATTERN.sub('===代码开始===', text)
    text = CODE_END_PATTERN.sub('===代码结束===', text)
    
    # 补充章节标记（如"1.1"改为"## 1.1"，便于分块识别）
    text = SECTION_PATTERN.sub(r'## \1 ', text)
    text = CHAPTER_PATTERN.sub(r'# \1 ', text)
    
    return text


def process_single_file(filename: str, input_dir: str, output_dir: str) -> Optional[Dict]:
    """处理单个文件（用于并行处理）"""
    try:
        input_path = os.path.join(input_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件 {input_path} 不存在")
        
        # 读取原始文本（显式指定编码）
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        # 执行清洗流程
        cleaned = initial_clean(raw_text)
        no_redundant = remove_redundant_content(cleaned)
        standardized = standardize_terms_and_format(no_redundant)
        
        # 保存结果
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(standardized)
        
        # 计算压缩率，避免除以零错误
        reduction_ratio = 0.0
        if raw_text:
            reduction_ratio = 1 - len(standardized) / len(raw_text)
        
        # 返回日志数据
        return {
            "filename": filename,
            "raw_length": len(raw_text),
            "processed_length": len(standardized),
            "reduction_ratio": reduction_ratio,
            "status": "success"
        }
    
    except Exception as e:
        error_msg = f"处理文件 {filename} 时出错: {str(e)}"
        print(error_msg)
        traceback.print_exc()  # 添加详细的错误堆栈信息
        return {
            "filename": filename,
            "status": "error",
            "error_message": str(e),
            "traceback": traceback.format_exc()  # 记录完整的错误堆栈
        }


def batch_process_texts(input_dir: str, output_dir: str, use_parallel: bool = True, max_workers: int = None) -> None:
    """批量处理目录下的所有文本文件（添加了并行处理选项）"""
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录 {input_dir} 不存在，请检查路径是否正确")
    
    os.makedirs(output_dir, exist_ok=True)
    log = []  # 记录处理日志
    
    # 获取所有文本文件
    text_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    if not text_files:
        print(f"警告：在 {input_dir} 目录中未找到文本文件")
        return
    
    print(f"找到 {len(text_files)} 个文本文件需要处理...")
    
    # 确定工作线程数（根据CPU核心数自动调整）
    if max_workers is None:
        max_workers = min(32, multiprocessing.cpu_count() + 4)
    
    # 使用并行处理或顺序处理
    if use_parallel and len(text_files) > 1:
        print(f"使用 {max_workers} 个工作线程进行并行处理...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用map方法正确处理结果
            futures = list(tqdm(
                executor.map(lambda f: process_single_file(f, input_dir, output_dir), text_files),
                total=len(text_files),
                desc="并行处理文本"
            ))
            # 收集结果
            for result in futures:
                if result:
                    log.append(result)
    else:
        # 顺序处理（小文件集或调试时使用）
        for filename in tqdm(text_files, desc="处理文本"):
            result = process_single_file(filename, input_dir, output_dir)
            if result:
                log.append(result)
    
    # 保存日志
    try:
        log_path = os.path.join(output_dir, "processing_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"处理日志已保存至 {log_path}")
    except Exception as e:
        print(f"保存日志时出错: {str(e)}")
    
    # 统计信息
    success_count = sum(1 for entry in log if entry.get('status') == 'success')
    error_count = len(log) - success_count
    print(f"批量处理完成：成功 {success_count}/{len(text_files)} 个文件，"
          f"失败 {error_count} 个文件，结果保存至 {output_dir}")


def cleaning():
    """供main.py调用的清洗函数"""
    print("="*50)
    print("文档清洗处理系统")
    print("="*50)
    
    # 配置参数
    INPUT_DIR = "processed_pdfs"  # 输入目录（存放提取的原始文本）
    OUTPUT_DIR = "standardized_texts"  # 输出目录（存放清洗后的文本）
    USE_PARALLEL = True  # 启用并行处理
    MAX_WORKERS = None   # 自动确定工作线程数
    
    # 执行批量处理
    batch_process_texts(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        use_parallel=USE_PARALLEL,
        max_workers=MAX_WORKERS
    )
    
    print("\n处理完成！")
    print(f"清洗后的文本已保存至: {os.path.abspath(OUTPUT_DIR)}")
    print(f"处理日志已生成: {os.path.join(OUTPUT_DIR, 'processing_log.json')}")
    print("="*50)

