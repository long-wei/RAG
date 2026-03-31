# PDF 文件转换为纯文本
import os
import json
import re
from fnmatch import fnmatch
import fitz

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
    """异常处理和日志记录"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        # 提前初始化page_num防止未定义错误
        page_num = 0

        with fitz.open(pdf_path) as pdf:
            full_text = ""
            pages = []
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                pages.append(page)
                text = page.get_text("text")
                full_text += text + "\n"

            formulas = extract_formulas(pages)
            tables = extract_tables_to_markdown(pdf)

        text_with_code = extract_code_blocks(full_text)

        base_name = os.path.basename(pdf_path)
        # 文件写入
        with open(os.path.join(output_dir, f"{base_name}.formulas.json"), "w", encoding='utf-8') as f:
            json.dump(formulas, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, f"{base_name}.tables.md"), "w", encoding="utf-8") as f:
            for t in tables:
                f.write(f"### 第{t['page']}页 表格{t['table_idx']}\n{t['markdown']}\n\n")

        with open(os.path.join(output_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
            f.write(text_with_code)

        return True

    except Exception as e:
        # 确保page_num已定义
        error_msg = f"处理失败：{pdf_path}，错误：{str(e)} 在页码：{page_num + 1 if 'page_num' in locals() else '未知'}"
        print(error_msg)
        with open("processing_errors.log", "a", encoding="utf-8") as f:
            f.write(error_msg + "\n")
        return False


# 公式处理
def extract_formulas(pages):
    """提取PDF中的公式并保留原始格式"""
    formulas = []
    for page_num, page in enumerate(pages):
        text = page.get_text("text")
        if not text:
            continue
        # 正则匹配公式（包含$、_、^等符号的连续文本）
        formula_pattern = r'\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)'  # 匹配$...$、\[...\]、\(...\)格式
        page_formulas = re.findall(formula_pattern, text, re.DOTALL)
        for formula in page_formulas:
            # 清理公式中的多余空格
            cleaned_formula = re.sub(r'\s+', ' ', formula).strip()
            formulas.append({
                "page": page_num + 1,
                "formula": cleaned_formula
            })
    return formulas


# 代码块处理
def extract_code_blocks(text):
    """代码块识别"""
    if not text:
        return ""

    # 代码块识别
    code_pattern = re.compile(
        r'(?:^|\n)(?:\s{4}|\t)(?:.*\n?)+?(?=\n(?:\s{0,3}(?![ \t])|\Z))',
        re.DOTALL | re.MULTILINE
    )

    def add_markers(match):
        code = re.sub(r'\n{3,}', '\n\n', match.group(0).strip())
        return f"\n===代码开始===\n{code}\n===代码结束===\n"

    # 分阶段处理
    text = code_pattern.sub(add_markers, text)

    # 行内代码处理
    return re.sub(r'`([^`\n]+)`', r'`\1`', text)


# 表格处理
def extract_tables_to_markdown(pdf):
    """使用PyMuPDF表格提取"""
    tables_md = []
    for page_num, page in enumerate(pdf):
        try:
            tabs = page.find_tables()  # PyMuPDF内置表格识别
            for tab_idx, tab in enumerate(tabs):
                # 转换为Markdown表格
                table = tab.extract()
                if not table:
                    continue

                md_table = []
                header = table[0]
                md_table.append("| " + " | ".join(str(cell or "") for cell in header) + " |")
                md_table.append("| " + " | ".join(["---"] * len(header)) + " |")

                for row in table[1:]:
                    md_table.append("| " + " | ".join([str(cell or "") for cell in row]) + " |")

                tables_md.append({
                    "page": page_num + 1,
                    "table_idx": tab_idx,
                    "markdown": "\n".join(md_table)
                })
        except Exception as e:
            print(f"警告：第{page_num + 1}页表格提取失败: {str(e)}")
            continue
    return tables_md


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