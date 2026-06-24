"""
数据处理流程测试脚本
测试 PDF 提取 → 数据清洗 → 文档分块 的完整流程
"""
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from Replaced_text import Replaced
from Data_cleaning import cleaning
from Document_chunking import chunking


def test_pipeline():
    """测试完整的数据处理流水线"""
    print("="*70)
    print("数据处理流程测试")
    print("="*70)
    
    # 检查输入文件
    dataset_dir = Path("Dataset")
    if not dataset_dir.exists() or not list(dataset_dir.glob("*.pdf")):
        print("❌ 错误：Dataset 目录中没有找到 PDF 文件")
        print("请将 PDF 文件放入 Dataset 目录后再运行测试")
        return False
    
    pdf_count = len(list(dataset_dir.glob("*.pdf")))
    print(f"✅ 找到 {pdf_count} 个 PDF 文件\n")
    
    try:
        # 步骤 1: PDF 提取
        print("\n" + "="*70)
        print("步骤 1/3: PDF 内容提取 (Replaced_text.py)")
        print("="*70)
        Replaced()
        
        # 检查提取结果
        processed_dir = Path("processed_pdfs")
        if not processed_dir.exists() or not list(processed_dir.glob("*.txt")):
            print("❌ 错误：PDF 提取失败，未生成文本文件")
            return False
        
        txt_count = len(list(processed_dir.glob("*.txt")))
        print(f"✅ 成功提取 {txt_count} 个文本文件")
        
        # 步骤 2: 数据清洗
        print("\n" + "="*70)
        print("步骤 2/3: 数据清洗与标准化 (Data_cleaning.py)")
        print("="*70)
        cleaning()
        
        # 检查清洗结果
        standardized_dir = Path("standardized_texts")
        if not standardized_dir.exists() or not list(standardized_dir.glob("*.txt")):
            print("❌ 错误：数据清洗失败，未生成标准化文本")
            return False
        
        std_count = len(list(standardized_dir.glob("*.txt")))
        print(f"✅ 成功清洗 {std_count} 个文本文件")
        
        # 步骤 3: 文档分块
        print("\n" + "="*70)
        print("步骤 3/3: 语义分块 (Document_chunking.py)")
        print("="*70)
        chunking()
        
        # 检查分块结果
        chunks_dir = Path("semantic_chunks")
        if not chunks_dir.exists() or not list(chunks_dir.glob("*_chunks.jsonl")):
            print("❌ 错误：文档分块失败，未生成分块文件")
            return False
        
        chunk_count = len(list(chunks_dir.glob("*_chunks.jsonl")))
        print(f"✅ 成功生成 {chunk_count} 个分块文件")
        
        # 总结
        print("\n" + "="*70)
        print("✅ 测试完成！所有步骤执行成功")
        print("="*70)
        print(f"\n输出文件统计:")
        print(f"  - 原始提取: processed_pdfs/ ({txt_count} 个文件)")
        print(f"  - 清洗后文本: standardized_texts/ ({std_count} 个文件)")
        print(f"  - 语义分块: semantic_chunks/ ({chunk_count} 个文件)")
        print(f"\n可以查看以下日志文件了解详细信息:")
        print(f"  - processed_pdfs/processing_log.json")
        print(f"  - standardized_texts/processing_log.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
