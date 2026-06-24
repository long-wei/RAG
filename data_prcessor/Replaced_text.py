# PDF 文件转换为纯文本 - 使用 pymupdf4llm
import os
import json
import re
from fnmatch import fnmatch
import pymupdf4llm

def find_pdf_files(directory):
    """
    directory:pdf所在文件夹路径
    :param directory:
    :return: pdf_files（包含所有pdf文件路径的列表）
    """
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch(file, '*.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def process_pdf(pdf_path, output_dir):
    """使用 pymupdf4llm 进行高级 PDF 提取（仅提取，不做清洗）"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(pdf_path)
        
        # 使用 pymupdf4llm 提取为 Markdown 格式（保留原始结构）
        md_text = pymupdf4llm.to_markdown(
            pdf_path,
            show_progress=False,
            write_images=False,  # 不提取图片
            image_format="png",
            embed_images=False,
            page_chunks=True  # 按页分块
        )
        
        # 如果返回的是字典列表（page_chunks=True）
        if isinstance(md_text, list):
            full_text = ""
            all_formulas = []
            all_tables = []
            
            for page_data in md_text:
                page_num = page_data.get("metadata", {}).get("page", 0) + 1
                text = page_data.get("text", "")
                full_text += text + "\n\n"
                
                # 提取公式（Markdown 中的 LaTeX 格式）
                formulas_in_page = extract_formulas_from_markdown(text, page_num)
                all_formulas.extend(formulas_in_page)
                
                # 提取表格（Markdown 表格格式）
                tables_in_page = extract_tables_from_markdown(text, page_num)
                all_tables.extend(tables_in_page)
        else:
            # 如果是字符串
            full_text = md_text
            all_formulas = extract_formulas_from_markdown(full_text, 0)
            all_tables = []
        
        # 注意：这里不做文本清洗，直接保存原始提取结果
        # 文本清洗由 Data_cleaning.py 负责
        
        # 文件写入
        with open(os.path.join(output_dir, f"{base_name}.formulas.json"), "w", encoding='utf-8') as f:
            json.dump(all_formulas, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, f"{base_name}.tables.md"), "w", encoding="utf-8") as f:
            for t in all_tables:
                f.write(f"### 第{t['page']}页 表格{t['table_idx']}\n{t['markdown']}\n\n")

        with open(os.path.join(output_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)  # 保存原始提取的文本

        return True

    except Exception as e:
        error_msg = f"处理失败：{pdf_path}，错误：{str(e)}"
        print(error_msg)
        with open("processing_errors.log", "a", encoding="utf-8") as f:
            f.write(error_msg + "\n")
        return False


# 从 Markdown 中提取公式（辅助函数，供后续处理使用）
def extract_formulas_from_markdown(text, page_num):
    """从 Markdown 文本中提取 LaTeX 公式"""
    formulas = []
    if not text:
        return formulas
    
    # 匹配行内公式 $...$ 和行间公式 $$...$$、\[...\]、\(...\)
    formula_patterns = [
        r'\$\$(.+?)\$\$',  # $$...$$
        r'\\\[(.+?)\\\]',  # \[...\]
        r'\\\((.+?)\\\)',  # \(...\)
        r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)'  # $...$ (非贪婪)
    ]
    
    formula_idx = 0
    for pattern in formula_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for formula_content in matches:
            cleaned_formula = re.sub(r'\s+', ' ', formula_content).strip()
            if cleaned_formula:  # 只添加非空公式
                formulas.append({
                    "page": page_num,
                    "formula_idx": formula_idx,
                    "formula": f"${cleaned_formula}$" if '$$' not in formula_content else f"$${cleaned_formula}$$"
                })
                formula_idx += 1
    
    return formulas


# 从 Markdown 中提取表格（辅助函数，供后续处理使用）
def extract_tables_from_markdown(text, page_num):
    """从 Markdown 文本中提取表格"""
    tables = []
    if not text:
        return tables
    
    # 匹配 Markdown 表格（至少包含表头分隔符 |---|）
    table_pattern = r'(\|(?:[^|]+\|)+\s*\n\|(?:[-:|\s]+\|)+\s*\n(?:\|(?:[^|]+\|)+\s*\n?)*)'
    
    matches = re.finditer(table_pattern, text, re.MULTILINE)
    
    for tab_idx, match in enumerate(matches):
        table_md = match.group(1).strip()
        if table_md:  # 只添加非空表格
            tables.append({
                "page": page_num,
                "table_idx": tab_idx,
                "markdown": table_md
            })
    
    return tables


def Replaced():
    # 确保Dataset目录存在
    if not os.path.exists("Dataset"):
        os.makedirs("Dataset")
        print("请将PDF文件放入Dataset目录中")
    else:
        pdf_files = find_pdf_files("Dataset")
        if not pdf_files:
            print("在Dataset目录中未找到PDF文件，请添加PDF文件后再运行")
        else:
            # 创建processed_pdfs目录
            os.makedirs("processed_pdfs", exist_ok=True)
            # 批量处理所有PDF
            print(f"找到 {len(pdf_files)} 个PDF文件，开始处理...")
            success_count = 0
            for i, pdf in enumerate(pdf_files, 1):
                print(f"正在处理 ({i}/{len(pdf_files)}): {os.path.basename(pdf)}")
                if process_pdf(pdf, output_dir="processed_pdfs"):
                    success_count += 1

            print(f"处理完成 {success_count}/{len(pdf_files)} 个文件")

if __name__ == "__main__":
    Replaced()