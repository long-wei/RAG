# 自然语言处理技术文档问答系统

一个基于检索增强生成（RAG）的智能问答系统，专门用于回答**自然语言处理（NLP）**和深度学习技术文档相关问题。系统提供**Web 界面**和**GUI 桌面应用**两种交互方式，支持文本输入、语音识别和语音合成回答。

## 目录结构

```
深度学习课程设计/
├── data_prcessor/           # 数据处理模块
│   ├── Dataset/             # 原始 PDF 文件目录（9 个章节的 NLP 与深度学习教材）
│   ├── processed_pdfs/      # PDF 提取后的文本、公式、表格
│   ├── standardized_texts/  # 标准化处理后的文本
│   ├── semantic_chunks/     # 语义分块结果
│   ├── main0.py             # 数据处理流程总入口
│   ├── Replaced_text.py     # PDF 文本、公式、表格提取
│   ├── Data_cleaning.py     # 文本清洗和标准化
│   ├── Document_chunking.py # 文档语义分块
│   └── Text_vectorization.py# 文本向量化并构建 FAISS 索引
├── models/                  # 预训练向量模型（paraphrase-multilingual-mpnet-base-v2）
├── sound_model/             # Vosk 语音识别模型（中文）
├── static/                  # Web 静态资源
│   ├── css/
│   └── js/
├── templates/               # Web 模板文件
│   ├── base.html
│   ├── chat.html
│   └── index.html
├── index_loader.py          # 索引和模型加载模块
├── query_processor.py       # 查询预处理和向量化模块
├── retrieval_optimization.py # 检索优化模块
├── LLM_Integration.py       # 阿里云百炼大语言模型集成
├── Evaluation_feedback.py   # 评估与反馈模块
├── web_app.py               # Web 应用主程序入口
├── main.py                  # GUI 桌面应用主程序入口
└── test_api.py              # API 测试脚本
```

## 系统功能

### 1. **双模式交互界面**：
   - **Web 应用**：基于 Flask 的 Web 界面，支持浏览器访问
   - **GUI 桌面应用**：基于 Tkinter 的桌面客户端，跨平台支持

### 2. **智能问答核心**：
   - 文本输入查询
   - 语音输入识别（Vosk 离线语音识别）
   - 语音合成回答（pyttsx3）
   - 检索增强生成（RAG）机制
   - 上下文感知回答生成

### 3. **文档处理流程**：
   - PDF 文档提取（文本、公式、表格）
   - 文本清洗与标准化
   - 语义分块处理
   - 文本向量化与 FAISS 索引构建

### 4. **系统评估与反馈**：
   - 检索质量评估（准确率、召回率、MRR）
   - 回答质量评估（ROUGE-L、BLEU）
   - 用户评分和反馈收集与分析

## 技术架构

### 核心组件
- **文档处理**：PyMuPDF（PDF 解析）、正则表达式处理（公式和代码提取）
- **文本向量化**：Transformers 库、paraphrase-multilingual-mpnet-base-v2 模型（本地部署）
- **相似度检索**：FAISS 向量搜索引擎（Facebook AI Similarity Search）
- **大语言模型**：阿里云百炼 API（Qwen 模型）
- **语音识别**：Vosk 离线语音识别引擎（支持中文）
- **语音合成**：pyttsx3 文本转语音引擎
- **Web 框架**：Flask（后端服务）
- **GUI 框架**：Tkinter（桌面客户端）

### 工作流程

#### 1. **数据预处理阶段**：
   ```bash
   cd data_prcessor
   python main0.py  # 一键执行完整流程
   ```
   详细步骤：
   - PDF 文档解析提取文本、公式和表格
   - 文本清洗、标准化和术语统一
   - 语义分块处理，保留代码和公式结构
   - 文本向量化并构建 FAISS 索引
   - 生成 `tech_doc_index.index` 和 `tech_doc_index_metadata.jsonl`

#### 2. **问答处理阶段**：

**Web 方式**：
   ```bash
   python web_app.py
   ```
   - 用户通过 Web 界面输入问题
   - 查询预处理和向量化
   - FAISS 检索相关文档片段
   - 检索结果优化（去重、过滤等）
   - 构建 Prompt 并调用大语言模型生成回答
   - 通过 Web 界面展示回答和相关文档片段

**GUI 桌面方式**：
   ```bash
   python main.py
   ```
   - 支持文本输入和语音输入两种方式
   - 实时语音识别（Vosk 模型）
   - 语音合成播放回答
   - 图形化界面显示对话历史
   - 支持评分和反馈提交

#### 3. **评估与反馈阶段**：
   - 自动评估检索和回答质量
   - 收集用户评分和反馈
   - 分析反馈数据以优化系统
   - 生成实验结果文件（`experiment_results.jsonl` 等）

## 安装与运行

详细环境配置请参考 [ENVIRONMENT.md](ENVIRONMENT.md) 文件。

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载 NLTK 数据
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. 配置 API 密钥
设置阿里云百炼 API 密钥环境变量：
```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key_here"

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

### 4. 构建文档索引
将 PDF 文档放入 `data_prcessor/Dataset/` 目录后运行：
```bash
cd data_prcessor
python main0.py
```

### 5. 运行系统

**方式一：Web 应用**
```bash
python web_app.py
```
访问：http://localhost:5000

**方式二：GUI 桌面应用**
```bash
python main.py
```

### 6. 测试 API
```bash
python test_api.py
```

## 数据处理流程详解

### 第一步：PDF 内容提取
```bash
cd data_prcessor
python Replaced_text.py
```
- 从 `Dataset/` 目录读取 9 个章节的深度学习教材 PDF
- 提取文本、公式（保存为 `.formulas.json`）、表格（保存为 `.tables.md`）
- 输出到 `processed_pdfs/` 目录

### 第二步：文本清洗和标准化
```bash
python Data_cleaning.py
```
- 清理特殊字符和冗余空格
- 统一术语表达（如"词向量"→"词向量（Word Vector）"）
- 输出到 `standardized_texts/` 目录

### 第三步：文档语义分块
```bash
python Document_chunking.py
```
- 根据语义完整性进行分块
- 保留公式、代码块的完整性
- 输出到 `semantic_chunks/` 目录（JSONL 格式）

### 第四步：文本向量化与索引构建
```bash
python Text_vectorization.py
```
- 使用 paraphrase-multilingual-mpnet-base-v2 模型生成向量
- 构建 FAISS 向量索引
- 生成 `tech_doc_index.index` 和 `tech_doc_index_metadata.jsonl`

### 一键执行全部流程
```bash
cd data_prcessor
python main0.py  # 自动依次执行上述所有步骤
```

## 项目特点

- 🎯 **垂直领域问答**：针对自然语言处理（NLP）技术文档优化的专业问答系统
- 💻 **双模式界面**：Web 应用 + GUI 桌面客户端，满足不同使用场景
- 🔍 **检索增强生成**：结合 FAISS 精确检索和大语言模型生成能力
- 🎤 **多模态交互**：支持文本输入、语音输入、语音合成多种交互方式
- 📊 **质量评估体系**：完整的自动化评估指标和用户反馈机制
- 🔧 **模块化架构**：清晰的功能模块划分，易于扩展和维护
- 📚 **九章教材覆盖**：涵盖从绪论、NLP 基础、工具集、神经网络、词向量到预训练模型的完整知识体系

## 注意事项

### API 配置
- 需要注册阿里云百炼平台账号并创建 API 密钥
- 设置 `DASHSCOPE_API_KEY` 环境变量

### 语音功能
- 语音识别使用 Vosk 离线引擎，无需联网
- 语音合成使用 pyttsx3，支持离线运行
- GUI 应用需要麦克风设备支持

### 性能要求
- 建议配置：8GB+ 内存，GPU 加速（可选）
- 首次加载模型和索引需要一定时间
- Web 应用默认端口 5000，可通过参数修改

### 浏览器兼容性（Web 模式）
- 推荐使用 Chrome、Firefox、Edge 等现代浏览器
- 本地开发环境使用 HTTP 协议即可

### 模型和索引
- 向量模型已预置在 `models/` 目录
- 语音识别模型已预置在 `sound_model/` 目录
- 需先运行数据处理流程生成文档索引才能使用问答功能