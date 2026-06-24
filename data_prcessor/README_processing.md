# 数据处理模块说明

## 📋 模块职责

本目录包含 RAG 系统的完整数据处理流水线，采用模块化设计，各模块职责清晰分离。

### 处理流程图

```
PDF 原始文件
    ↓
┌─────────────────────────┐
│  Replaced_text.py       │  ← PDF 内容提取（不做清洗）
│  - pymupdf4llm 提取      │
│  - 公式识别              │
│  - 表格提取              │
└─────────────────────────┘
    ↓
processed_pdfs/*.txt (原始提取结果)
    ↓
┌─────────────────────────┐
│  Data_cleaning.py       │  ← 数据清洗与标准化
│  - 特殊字符清理          │
│  - Markdown 格式处理     │
│  - 术语统一              │
│  - 代码块格式转换        │
└─────────────────────────┘
    ↓
standardized_texts/*.txt (清洗后文本)
    ↓
┌─────────────────────────┐
│  Document_chunking.py   │  ← 语义分块
│  - 基于标题层级分块      │
│  - 代码块完整性保护      │
│  - 公式完整性验证        │
└─────────────────────────┘
    ↓
semantic_chunks/*_chunks.jsonl (最终分块结果)
```

---

## 🔧 模块详细说明

### 1. Replaced_text.py - PDF 内容提取

**职责**: 将 PDF 文件转换为结构化文本，不做任何清洗操作

**核心功能**:
- ✅ 使用 `pymupdf4llm` 进行高级 PDF 提取
- ✅ 自动识别并保留文档结构（Markdown 格式）
- ✅ 提取 LaTeX 公式并保存为 JSON
- ✅ 提取表格并保存为 Markdown
- ✅ 按页分块处理大文件

**输出文件**:
- `*.txt` - 原始提取的文本（包含 Markdown 标记）
- `*.formulas.json` - 提取的公式列表
- `*.tables.md` - 提取的表格（Markdown 格式）

**使用方法**:
```python
from Replaced_text import Replaced
Replaced()  # 处理 Dataset/ 目录下的所有 PDF
```

**关键特性**:
- 🎯 **单一职责**: 只负责提取，不修改内容
- 📊 **保留原始数据**: 便于后续多次实验和对比
- 🔍 **详细日志**: 记录每个文件的处理状态

---

### 2. Data_cleaning.py - 数据清洗与标准化

**职责**: 对原始提取的文本进行清洗、标准化和格式转换

**核心功能**:
- ✅ 去除冗余符号（保留 Markdown 标记）
- ✅ 处理 Markdown 图片和链接
- ✅ 合并多余空行（保留段落间距）
- ✅ 去除技术文档冗余内容（目录、小结等）
- ✅ 术语统一（中英文对照）
- ✅ 代码块格式转换（``` → ===代码开始===）
- ✅ 章节标记规范化

**输出文件**:
- `*.txt` - 清洗后的标准化文本
- `processing_log.json` - 处理日志（包含压缩率等统计）

**使用方法**:
```python
from Data_cleaning import cleaning
cleaning()  # 处理 processed_pdfs/ 目录下的所有文本
```

**关键改进**:
- 🎨 **Markdown 支持**: 完整保留标题、列表、代码块等格式
- 🔄 **格式转换**: 自动将 Markdown 代码块转换为统一标记
- ⚡ **并行处理**: 支持多线程加速批量处理
- 📈 **质量监控**: 记录每文件的压缩率和处理状态

---

### 3. Document_chunking.py - 语义分块

**职责**: 将清洗后的文本按语义逻辑分割成适合检索的块

**核心功能**:
- ✅ 基于标题层级的智能分块
- ✅ 代码块完整性保护（不拆分代码）
- ✅ 公式完整性验证
- ✅ 块间重叠保留上下文
- ✅ 生成带元数据的 JSONL 文件

**输出文件**:
- `*_chunks.jsonl` - 分块结果（每行一个 JSON 对象）

**分块策略**:
```python
chunk_size = 700      # 单块最大字符数
chunk_overlap = 100   # 块间重叠字符数

分隔符优先级:
1. \n#    - 章节标题
2. \n##   - 小节标题
3. \n###  - 子小节标题
4. \n\n   - 段落分隔
5. \n     - 换行
6. . / 。 - 句号
7. , / ， - 逗号
```

**使用方法**:
```python
from Document_chunking import chunking
chunking()  # 处理 standardized_texts/ 目录下的所有文本
```

---

## 🚀 快速开始

### 方式 1: 使用测试脚本（推荐）

```bash
cd data_prcessor
python test_pipeline.py
```

测试脚本会自动执行完整的处理流程，并验证每一步的结果。

### 方式 2: 手动执行各步骤

```python
# 步骤 1: PDF 提取
from Replaced_text import Replaced
Replaced()

# 步骤 2: 数据清洗
from Data_cleaning import cleaning
cleaning()

# 步骤 3: 文档分块
from Document_chunking import chunking
chunking()
```

### 方式 3: 使用主入口

```bash
python main0.py
```

---

## 📁 目录结构

```
data_prcessor/
├── Dataset/                    # 输入：PDF 文件目录
│   ├── ch1-绪论_25.pdf
│   ├── ch2-自然语言处理基础_25.pdf
│   └── ...
│
├── processed_pdfs/             # 中间产物：原始提取结果
│   ├── *.txt                   # 原始文本（含 Markdown）
│   ├── *.formulas.json         # 公式列表
│   ├── *.tables.md             # 表格（Markdown）
│   └── processing_errors.log   # 错误日志
│
├── standardized_texts/         # 中间产物：清洗后文本
│   ├── *.txt                   # 标准化文本
│   └── processing_log.json     # 处理日志
│
├── semantic_chunks/            # 输出：最终分块结果
│   └── *_chunks.jsonl          # 分块文件
│
├── Replaced_text.py            # PDF 提取模块
├── Data_cleaning.py            # 数据清洗模块
├── Document_chunking.py        # 文档分块模块
├── Text_vectorization.py       # 文本向量化（下一步）
├── test_pipeline.py            # 流程测试脚本
└── README_processing.md        # 本文档
```

---

## ⚙️ 配置参数

### Replaced_text.py

```python
# PDF 提取配置
md_text = pymupdf4llm.to_markdown(
    pdf_path,
    show_progress=False,      # 是否显示进度条
    write_images=False,       # 是否提取图片
    image_format="png",       # 图片格式
    embed_images=False,       # 是否嵌入图片
    page_chunks=True          # 按页分块
)
```

### Data_cleaning.py

```python
# 清洗配置
INPUT_DIR = "processed_pdfs"      # 输入目录
OUTPUT_DIR = "standardized_texts" # 输出目录
USE_PARALLEL = True               # 启用并行处理
MAX_WORKERS = None                # 自动确定线程数
```

### Document_chunking.py

```python
# 分块配置
chunk_size = 700          # 单块字符数
chunk_overlap = 100       # 重叠字符数
separators = [...]        # 分隔符优先级
```

---

## 🔍 常见问题

### Q1: 为什么要把提取和清洗分开？

**A**: 
- **职责清晰**: 每个模块只做一件事
- **便于调试**: 可以定位问题在哪个阶段
- **保留原始数据**: 支持多次清洗实验
- **可复用性**: 提取模块可用于其他场景

### Q2: 如何处理大型 PDF 文件？

**A**:
- `pymupdf4llm` 已支持按页分块处理
- 数据清洗支持并行处理（默认启用）
- 建议分批处理超大文件（>200页）

### Q3: Markdown 格式会被保留吗？

**A**:
- ✅ 是的！清洗模块专门优化了 Markdown 支持
- 标题、列表、代码块、公式等都会保留
- 这些结构信息对 RAG 检索非常重要

### Q4: 如何查看处理日志？

**A**:
```bash
# 查看提取日志
cat processed_pdfs/processing_errors.log

# 查看清洗统计
cat standardized_texts/processing_log.json
```

### Q5: 分块效果不理想怎么办？

**A**:
调整 `Document_chunking.py` 中的参数：
- `chunk_size`: 增大可获得更完整的上下文
- `chunk_overlap`: 增大可减少信息丢失
- `separators`: 调整分隔符优先级

---

## 📊 性能参考

基于 9 个 NLP 教材 PDF（共约 500 页）的测试结果：

| 阶段 | 处理时间 | 输出文件大小 |
|------|---------|------------|
| PDF 提取 | ~2-3 分钟 | ~5 MB |
| 数据清洗 | ~10-20 秒 | ~4 MB |
| 文档分块 | ~5-10 秒 | ~8 MB (JSONL) |
| **总计** | **~3 分钟** | **~17 MB** |

*注：实际时间取决于文件大小和系统配置*

---

## 🛠️ 故障排除

### 问题 1: 找不到 PDF 文件

```
❌ 错误：Dataset 目录中没有找到 PDF 文件
```

**解决**: 将 PDF 文件放入 `data_prcessor/Dataset/` 目录

### 问题 2: pymupdf4llm 导入失败

```
ModuleNotFoundError: No module named 'pymupdf4llm'
```

**解决**:
```bash
pip install pymupdf4llm==1.27.2.3
```

### 问题 3: 内存不足

```
MemoryError: Unable to allocate...
```

**解决**:
- 减少 `MAX_WORKERS` 数量
- 分批处理大型 PDF
- 关闭图片提取（`write_images=False`）

---

## 📝 更新日志

### 2026-06-23
- ✨ 升级为使用 `pymupdf4llm` 进行 PDF 提取
- 🔄 分离 PDF 提取和数据清洗职责
- 🎨 增强 Markdown 格式支持
- 📊 新增流程测试脚本
- 📝 完善文档说明

---

## 🔗 相关文档

- [项目难题与解决方案](../docs/problems_and_solutions.md)
- [环境配置说明](../ENVIRONMENT.md)
- [项目 README](../README.md)

---

**最后更新**: 2026-06-23  
**维护者**: 项目开发团队
