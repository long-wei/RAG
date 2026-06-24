import sys
import os
from flask import Flask, render_template, request, jsonify, session
import threading
import time
from datetime import timedelta

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from index_loader import load_all_components
from query_processor import generate_query_embedding
from retrieval_optimization import retrieve_and_optimize
from LLM_Integration import build_tech_prompt, load_llm, generate_tech_answer
from Evaluation_feedback import RAGEvaluator, FeedbackCollector

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.secret_key = 'your-secret-key-here'  # 实际部署时应使用安全的密钥
app.permanent_session_lifetime = timedelta(minutes=30)

# 全局变量存储系统组件
tokenizer = None
vector_model = None
index = None
metadata = None
llm_generator = None
evaluator = RAGEvaluator()
feedback_collector = FeedbackCollector()

# 添加一个标志来跟踪系统是否已经初始化
system_initialized = False
initialization_lock = threading.Lock()

def initialize_system():
    """初始化系统组件"""
    global tokenizer, vector_model, index, metadata, llm_generator, system_initialized
    try:
        print("正在加载系统组件...")
        tokenizer, vector_model, index, metadata = load_all_components()
        print("正在加载语言模型...")
        llm_generator = load_llm()  # 使用阿里云百炼API
        system_initialized = True
        print("系统初始化完成！")
        return True
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return False

def ensure_initialized():
    """确保系统已初始化"""
    global system_initialized
    if not system_initialized:
        with initialization_lock:
            if not system_initialized:
                initialize_system()

@app.route('/')
def index():
    """主页路由"""
    ensure_initialized()
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    ensure_initialized()

    data = request.get_json()
    user_query = data.get('query', '').strip()

    if not user_query:
        return jsonify({'error': '查询不能为空'}), 400

    try:
        # 生成查询向量
        query_emb = generate_query_embedding(user_query, tokenizer, vector_model)

        # 检索并优化相关分块
        retrieved_chunks = retrieve_and_optimize(
            query_embedding=query_emb,
            index=index,
            metadata=metadata,
            query_text=user_query
        )

        if not retrieved_chunks:
            answer = "抱歉，未找到与您的问题相关的技术内容。"
            return jsonify({
                'answer': answer,
                'chunks': [],
                'query': user_query
            })

        # 构建提示
        prompt = build_tech_prompt(user_query, retrieved_chunks)

        # 生成回答 (通过阿里云百炼API)
        answer = generate_tech_answer(prompt, llm_generator)

        # 保存到session供反馈使用
        session['current_query'] = user_query
        session['current_answer'] = answer
        session['current_chunks'] = [{
            'content': chunk.get('content', ''),
            'source': chunk.get('source', '未知'),
            'distance': chunk.get('distance', 0)
        } for chunk in retrieved_chunks]

        # 格式化检索到的chunks
        formatted_chunks = []
        for chunk in retrieved_chunks:
            formatted_chunks.append({
                'content': chunk.get('content', '')[:200] + '...' if len(chunk.get('content', '')) > 200 else chunk.get('content', ''),
                'source': chunk.get('source', '未知'),
                'distance': chunk.get('distance', 0)
            })

        return jsonify({
            'answer': answer,
            'chunks': formatted_chunks,
            'query': user_query
        })

    except Exception as e:
        error_msg = f"处理查询时出错: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """收集用户反馈"""
    ensure_initialized()

    data = request.get_json()
    rating = data.get('rating', 0)

    try:
        # 从session获取当前对话信息
        query = session.get('current_query', '')
        answer = session.get('current_answer', '')
        chunks = session.get('current_chunks', [])

        if not query:
            return jsonify({'error': '没有可评价的对话'}), 400

        # 收集反馈
        feedback_data = feedback_collector.collect_feedback(
            query=query,
            answer=answer,
            retrieved_chunks=chunks,
            user_rating=rating,
            evaluation_metrics={}
        )

        return jsonify({
            'message': '反馈已收集',
            'feedback_id': feedback_data['feedback_id']
        })

    except Exception as e:
        error_msg = f"收集反馈时出错: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    # 在应用启动时就开始初始化系统
    init_thread = threading.Thread(target=initialize_system)
    init_thread.start()

    app.run(debug=True, host='0.0.0.0', port=5000)
