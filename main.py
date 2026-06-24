import sys
import os
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from typing import List, Dict
import json
import wave
import pyaudio

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from index_loader import load_all_components
from query_processor import generate_query_embedding
from retrieval_optimization import retrieve_and_optimize
from LLM_Integration import build_tech_prompt, load_llm, generate_tech_answer
from Evaluation_feedback import RAGEvaluator, FeedbackCollector

# 导入Vosk
from vosk import Model, KaldiRecognizer


class TechDocChatbot:
    def __init__(self):
        self.tokenizer = None
        self.vector_model = None
        self.index = None
        self.metadata = None
        self.llm_generator = None
        self.evaluator = RAGEvaluator()
        self.feedback_collector = FeedbackCollector()
        self.is_listening = False
        self.tts_engine = None  # 延迟初始化TTS引擎
        self.tts_thread = None  # 保存TTS线程引用
        self.tts_stop_flag = False  # TTS停止标志

        # 初始化Vosk模型
        self.vosk_model = None
        self.initialize_vosk_model()

    def initialize_vosk_model(self):
        """初始化Vosk中文模型（使用指定的本地模型路径）"""
        try:
            # 使用指定的本地模型路径
            model_path = "vosk-model-small-cn-0.22"  # vosk-model-small-cn-0.22模型
            
            # 检查本地模型是否存在且为目录
            if os.path.exists(model_path) and os.path.isdir(model_path):
                print(f"使用本地Vosk模型: {model_path}")
                self.vosk_model = Model(model_path)
            else:
                print(f"错误：在路径 '{model_path}' 未找到本地Vosk模型")
                print("请确保模型已下载并放置在正确位置")
                self.vosk_model = None
            
        except Exception as e:
            print(f"Vosk模型初始化失败: {e}")
            self.vosk_model = None

    def initialize_system(self):
        """初始化系统组件"""
        try:
            print("正在加载系统组件...")
            self.tokenizer, self.vector_model, self.index, self.metadata = load_all_components()
            print("正在加载语言模型...")
            self.llm_generator = load_llm()  # 使用阿里云百炼API
            print("系统初始化完成！")
            return True
        except Exception as e:
            print(f"系统初始化失败: {e}")
            return False

    def process_query(self, query: str) -> tuple:
        """处理用户查询"""
        try:
            # 生成查询向量
            query_emb = generate_query_embedding(query, self.tokenizer, self.vector_model)

            # 检索并优化相关分块
            retrieved_chunks = retrieve_and_optimize(
                query_embedding=query_emb,
                index=self.index,
                metadata=self.metadata,
                query_text=query
            )

            if not retrieved_chunks:
                answer = "抱歉，未找到与您的问题相关的技术内容。"
                return answer, []

            # 构建提示
            prompt = build_tech_prompt(query, retrieved_chunks)

            # 生成回答 (通过阿里云百炼API)
            answer = generate_tech_answer(prompt, self.llm_generator)

            return answer, retrieved_chunks

        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            print(error_msg)
            return error_msg, []

    def text_to_speech(self, text: str):
        """将文本转换为语音"""

        def speak():
            try:
                # 每次播放前重新初始化TTS引擎，避免状态残留问题
                self.tts_engine = pyttsx3.init()
                # 检查是否需要停止
                while not self.tts_stop_flag:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    break  # 只播放一次
            except Exception as e:
                print(f"语音播放出错: {e}")
            finally:
                # 确保清理TTS引擎状态
                if self.tts_engine:
                    try:
                        self.tts_engine.stop()
                    except:
                        pass

        # 在单独的线程中运行语音合成，避免阻塞GUI
        try:
            self.tts_stop_flag = False  # 重置停止标志
            self.tts_thread = threading.Thread(target=speak, daemon=True)
            self.tts_thread.start()
        except Exception as e:
            print(f"无法启动语音播放线程: {e}")

    def stop_tts(self):
        """停止当前的TTS播放"""
        self.tts_stop_flag = True
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass

    def speech_to_text(self):
        """使用Vosk进行语音识别"""
        if not self.vosk_model:
            return "语音识别模型未初始化"
            
        try:
            # 使用pyaudio录制音频
            p = pyaudio.PyAudio()
            
            # 创建音频流
            stream = p.open(format=pyaudio.paInt16,
                           channels=1,
                           rate=16000,
                           input=True,
                           frames_per_buffer=8192)
            
            # 创建识别器
            rec = KaldiRecognizer(self.vosk_model, 16000)
            
            print("请说话...")
            stream.start_stream()
            
            results = []
            # 录制大约5秒的音频
            for i in range(0, 100):  # 限制循环次数
                data = stream.read(4096, exception_on_overflow=False)
                if rec.AcceptWaveform(data):
                    words = json.loads(rec.Result())
                    results.append(words)
                    
                    # 如果有识别结果，跳出循环
                    if words.get("text", "").strip():
                        break
            
            # 获取最终结果
            words = json.loads(rec.FinalResult())
            results.append(words)
            
            # 关闭音频流
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 提取识别文本
            text = ""
            for result in results:
                if "text" in result:
                    text += result["text"] + " "
            
            final_text = text.strip()
            if not final_text:
                return "抱歉，我没有听清楚您说了什么。"
                
            print(f"识别结果: {final_text}")
            return final_text
            
        except Exception as e:
            return f"语音识别出错: {e}"

    def collect_feedback(self, query: str, answer: str, retrieved_chunks: List[Dict], rating: int):
        """收集用户反馈"""
        try:
            # 评估回答质量
            evaluation_metrics = {}
            if retrieved_chunks:
                # 这里可以添加更复杂的评估逻辑
                pass

            # 收集反馈
            feedback = self.feedback_collector.collect_feedback(
                query=query,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                user_rating=rating,
                evaluation_metrics=evaluation_metrics
            )
            print(f"反馈已收集: {feedback['feedback_id']}")
        except Exception as e:
            print(f"收集反馈时出错: {e}")


class ChatbotGUI:
    def __init__(self, chatbot: 'TechDocChatbot'):  # 修复类型提示
        self.chatbot = chatbot
        self.root = tk.Tk()
        self.root.title("技术文档问答系统")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="技术文档问答系统", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 10))

        # 输入框和按钮
        ttk.Label(main_frame, text="请输入您的问题:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))

        self.query_entry = ttk.Entry(main_frame, width=50)
        self.query_entry.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.query_entry.bind("<Return>", lambda event: self.send_query())

        send_button = ttk.Button(main_frame, text="发送", command=self.send_query)
        send_button.grid(row=2, column=2, padx=(5, 0), pady=(0, 10))
        
        stop_tts_button = ttk.Button(main_frame, text="停止播放", command=self.stop_tts)
        stop_tts_button.grid(row=2, column=3, padx=(5, 0), pady=(0, 10))

        # 语音输入按钮
        voice_button = ttk.Button(main_frame, text="语音输入", command=self.voice_input)
        voice_button.grid(row=3, column=0, pady=(0, 10))
        
        # 聊天显示区域
        self.chat_display = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=20)
        self.chat_display.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.chat_display.config(state=tk.DISABLED)

        # 评分区域
        rating_frame = ttk.Frame(main_frame)
        rating_frame.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(rating_frame, text="请对回答进行评分:").pack(side=tk.LEFT)

        self.rating_var = tk.IntVar(value=5)
        for i in range(1, 6):
            ttk.Radiobutton(rating_frame, text=str(i), variable=self.rating_var, value=i).pack(side=tk.LEFT)

        # 反馈按钮
        feedback_button = ttk.Button(main_frame, text="提交反馈", command=self.submit_feedback)
        feedback_button.grid(row=6, column=0, columnspan=4, pady=(0, 10))

        # 状态栏
        self.status_var = tk.StringVar(value="系统就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=7, column=0, columnspan=4, sticky=(tk.W, tk.E))

        # 存储当前对话信息
        self.current_query = ""
        self.current_answer = ""
        self.current_chunks = []

    def send_query(self):
        """发送文本查询"""
        query = self.query_entry.get().strip()
        if not query:
            return

        self.query_entry.delete(0, tk.END)
        self.display_message(f"您: {query}", "user")
        self.status_var.set("正在处理您的问题...")

        # 在单独的线程中处理查询，避免阻塞GUI
        threading.Thread(target=self.process_query, args=(query,), daemon=True).start()

    def voice_input(self):
        """语音输入"""
        self.display_message("正在录音，请说话...", "system")
        self.status_var.set("正在录音，请说话...")

        # 在单独的线程中处理语音输入，避免阻塞GUI
        threading.Thread(target=self.process_voice_input, daemon=True).start()

    def process_voice_input(self):
        """处理语音输入"""
        try:
            query = self.chatbot.speech_to_text()
            self.root.after(0, self.handle_voice_result, query)
        except Exception as e:
            self.root.after(0, self.display_message, f"语音识别出错: {e}", "system")
            self.root.after(0, self.status_var.set, "语音识别出错")

    def handle_voice_result(self, query):
        """处理语音识别结果"""
        self.display_message(f"识别结果: {query}", "system")
        if not query.startswith("抱歉") and not query.startswith("语音识别服务出错"):
            self.query_entry.delete(0, tk.END)
            self.query_entry.insert(0, query)
            self.send_query()
        else:
            self.status_var.set("语音识别完成")

    def process_query(self, query: str):
        """在后台线程中处理查询"""
        try:
            answer, chunks = self.chatbot.process_query(query)
            # 切换到主线程更新UI
            self.root.after(0, self.handle_query_result, query, answer, chunks)
        except Exception as e:
            self.root.after(0, self.display_message, f"处理查询时出错: {e}", "system")
            self.root.after(0, self.status_var.set, "处理完成")

    def handle_query_result(self, query: str, answer: str, chunks: List[Dict]):
        """处理查询结果"""
        self.display_message(f"助手: {answer}", "assistant")
        self.chatbot.text_to_speech(answer)

        # 保存当前对话信息
        self.current_query = query
        self.current_answer = answer
        self.current_chunks = chunks

        self.status_var.set("处理完成")

    def display_message(self, message: str, sender: str):
        """显示消息"""
        self.chat_display.config(state=tk.NORMAL)

        if sender == "user":
            tag = "user"
            self.chat_display.insert(tk.END, f"{message}\n", tag)
        elif sender == "assistant":
            tag = "assistant"
            self.chat_display.insert(tk.END, f"{message}\n", tag)
        else:
            tag = "system"
            self.chat_display.insert(tk.END, f"{message}\n", tag)

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def submit_feedback(self):
        """提交反馈"""
        if not self.current_query:
            self.display_message("暂无回答可评价", "system")
            return

        rating = self.rating_var.get()
        self.chatbot.collect_feedback(
            self.current_query,
            self.current_answer,
            self.current_chunks,
            rating
        )
        self.display_message(f"感谢您的评分: {rating}星", "system")

    def stop_tts(self):
        """停止当前的TTS播放"""
        self.chatbot.stop_tts()
        self.status_var.set("已停止语音播放")

    def run(self):
        """运行GUI"""
        # 初始化系统
        self.status_var.set("正在初始化系统...")
        if self.chatbot.initialize_system():
            self.status_var.set("系统初始化完成，可以开始提问了")
        else:
            self.status_var.set("系统初始化失败")

        self.root.mainloop()


def main():
    """主函数"""
    # 创建聊天机器人实例
    chatbot = TechDocChatbot()

    # 创建并运行GUI
    gui = ChatbotGUI(chatbot)
    gui.run()


if __name__ == "__main__":
    main()