# 深度学习课程设计 - 技术文档问答系统

一个基于检索增强生成（RAG）的智能问答系统，专门用于回答自然语言处理技术文档相关问题。系统提供了Web界面，支持文本输入和语音输出。

## 目录结构

```
深度学习课程设计/
├── data_prcessor/           # 数据处理模块
│   ├── Dataset/             # 原始PDF文件目录
│   ├── processed_pdfs/      # PDF提取后的文本、公式、表格
│   ├── standardized_texts/  # 标准化处理后的文本
│   ├── semantic_chunks/     # 语义分块结果
│   ├── Replaced_text.py     # PDF文本提取
│   ├── Data_cleaning.py     # 文本清洗和标准化
│   ├── Document_chunking.py # 文档语义分块
│   └── Text_vectorization.py# 文本向量化
├── models/                  # 预训练模型文件
├── sound_model/             # 语音识别模型
├── index_loader.py          # 索引加载模块
├── query_processor.py       # 查询处理模块
├── retrieval_optimization.py # 检索优化模块
├── LLM_Integration.py       # 大语言模型集成
├── Evaluation_feedback.py   # 评估与反馈模块
├── web_app.py               # Web应用主程序入口
└── test_api.py              # API测试脚本
```

## 系统功能

1. **文档处理流程**：
   - PDF文档提取（文本、公式、表格）
   - 文本清洗与标准化
   - 语义分块处理
   - 文本向量化与索引构建

2. **智能问答系统**：
   - 基于Web的交互界面
   - 文本输入支持
   - 语音合成回答
   - 检索增强生成（RAG）机制
   - 上下文感知回答生成

3. **系统评估与反馈**：
   - 检索质量评估（准确率、召回率、MRR）
   - 回答质量评估（ROUGE-L、BLEU）
   - 用户反馈收集与分析

## 技术架构

### 核心组件
- **文档处理**：PyMuPDF、正则表达式处理
- **文本向量化**：Transformers库、paraphrase-multilingual-mpnet-base-v2模型
- **相似度检索**：FAISS向量搜索引擎
- **大语言模型**：阿里云百炼API（Qwen模型）
- **语音处理**：Web Speech API（语音识别和合成）
- **用户界面**：Flask Web框架

### 工作流程
1. **数据预处理**：
   - PDF文档解析提取文本、公式和表格
   - 文本清洗、标准化和术语统一
   - 语义分块处理，保留代码和公式结构
   - 文本向量化并构建FAISS索引

2. **问答处理**：
   - 用户通过Web界面输入问题
   - 查询预处理和向量化
   - FAISS检索相关文档片段
   - 检索结果优化（去重、过滤等）
   - 构建Prompt并调用大语言模型生成回答
   - 通过Web界面展示回答和相关文档片段，同时支持语音播放

3. **评估与反馈**：
   - 自动评估检索和回答质量
   - 收集用户评分和反馈
   - 分析反馈数据以优化系统

## 安装与运行

请参考 [ENVIRONMENT.md](ENVIRONMENT.md) 文件了解详细环境配置说明。

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行系统
```bash
python web_app.py
```

## 数据处理流程

1. **PDF处理**：
   - 将PDF文件放入 `data_prcessor/Dataset/` 目录
   - 运行 `data_prcessor/Replaced_text.py` 提取内容

2. **文本清洗**：
   - 运行 `data_prcessor/Data_cleaning.py` 标准化文本

3. **语义分块**：
   - 运行 `data_prcessor/Document_chunking.py` 进行分块

4. **文本向量化**：
   - 运行 `data_prcessor/Text_vectorization.py` 生成向量索引

## 项目特点

- **专业技术问答**：专门针对深度学习技术文档设计
- **Web界面交互**：基于Web的用户界面，方便跨平台访问
- **检索增强生成**：结合文档检索和大语言模型生成
- **质量评估体系**：完整的系统评估和用户反馈机制
- **可扩展架构**：模块化设计，便于功能扩展和维护

## 注意事项

- 需要配置阿里云百炼API密钥
- 语音识别功能需要使用Chrome浏览器并通过HTTPS访问（生产环境）
- 系统对计算资源有一定要求，建议使用GPU加速