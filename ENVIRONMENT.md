# 环境配置说明

## 系统要求

- Python 3.8 或更高版本
- 建议使用支持CUDA的GPU以获得更好的性能（非必需）
- 至少8GB内存（推荐16GB或更高）

## 依赖安装

### 1. 基础依赖安装

```bash
pip install -r requirements.txt
```

### 2. NLTK数据下载

系统使用NLTK进行文本处理，需要下载相关数据：

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. PyAudio安装问题解决

在Windows系统上安装PyAudio可能会遇到问题，建议使用以下方法：

```bash
# 方法1：使用conda安装
conda install pyaudio

# 方法2：下载对应版本的wheel文件手动安装
# 访问 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio 下载对应版本
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```

注意：PyAudio主要用于语音识别功能，如果您只使用Web界面的文字交互功能，则可以跳过此步骤。

## 模型配置

### 1. 向量模型

系统使用预训练的paraphrase-multilingual-mpnet-base-v2模型进行文本向量化，模型文件位于 [models](models/) 目录中：
- `config.json`: 模型配置文件
- `pytorch_model.bin`: 模型权重文件
- `special_tokens_map.json`: 特殊标记映射文件
- `tokenizer_config.json`: 分词器配置
- `tokenizer.json`: 分词器模型文件

### 2. 语音识别模型

系统使用浏览器内置的Web Speech API进行语音识别，无需额外配置本地模型(GUI需要，web不需要)。

注意：语音识别功能需要满足以下条件：
1. 使用支持Web Speech API的浏览器（推荐Chrome）
2. 网站必须通过HTTPS协议访问（本地开发环境例外）
3. 用户需要授权麦克风访问权限

### 3. 大语言模型

系统通过阿里云百炼平台调用Qwen大语言模型：

1. 注册[阿里云百炼平台](https://help.aliyun.com/zh/bailian)账号
2. 创建API密钥
3. 设置环境变量：
   ```bash
   export DASHSCOPE_API_KEY=your_api_key_here
   ```
   或在代码中直接传入API密钥

## 索引构建

系统需要预先构建文档向量索引：

1. 将PDF文档放入 `data_prcessor/Dataset/` 目录
2. 运行数据处理脚本：
   ```bash
   cd data_prcessor
   python main0.py
   ```
   
或者依次运行以下脚本：
   ```bash
   cd data_prcessor
   python Replaced_text.py      # 提取PDF内容
   python Data_cleaning.py      # 清洗和标准化文本
   python Document_chunking.py  # 文档语义分块
   python Text_vectorization.py # 文本向量化并构建索引
   ```

处理完成后，会在项目根目录生成以下文件：
- `tech_doc_index.index`: FAISS向量索引文件
- `tech_doc_index_metadata.jsonl`: 文档元数据文件

## 运行系统

完成以上配置后，运行主程序：

```bash
python web_app.py
```

系统将启动Web服务器，默认端口为5000。可以通过浏览器访问 http://localhost:5000 使用问答系统。

## 常见问题

### 1. 模型加载失败
- 确认 [models](models/) 目录中的模型文件完整
- 检查是否有足够的磁盘空间和内存

### 2. 语音识别不可用
- 确保使用最新版Chrome浏览器
- 检查是否通过HTTPS访问网站（本地开发环境除外）
- 检查浏览器麦克风权限设置
- 确认操作系统麦克风权限已开启

### 3. API调用失败
- 确认已正确设置 `DASHSCOPE_API_KEY` 环境变量
- 检查网络连接是否正常

### 4. 索引文件缺失
- 确认已完成数据处理流程并生成索引文件
- 检查索引文件路径是否正确
