import sys
import os
import time
import json
from typing import List, Dict
import torch


# 添加项目路径（与主程序一致）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入主程序核心模块
from main import TechDocChatbot
from Evaluation_feedback import RAGEvaluator  # 复用评估器

# 1. 定义NLP领域50个分级测试问题（按报告中的测试集整理）
NLP_TEST_QUESTIONS: List[Dict] = [
    # 一、基础概念与入门（1-15题，简单）
    {"qid": "Q1", "difficulty": "easy", "question": "什么是自然语言处理（NLP）？"},
    {"qid": "Q2", "difficulty": "easy", "question": "自然语言处理的主要应用场景有哪些？"},
    {"qid": "Q3", "difficulty": "easy", "question": "什么是词法分析？它在NLP中的作用是什么？"},
    {"qid": "Q4", "difficulty": "easy", "question": "句法分析和语义分析的区别是什么？"},
    {"qid": "Q5", "difficulty": "easy", "question": "什么是停用词？为什么要去除停用词？"},
    {"qid": "Q6", "difficulty": "easy", "question": "什么是分词？中文分词和英文分词有何不同？"},
    {"qid": "Q7", "difficulty": "easy", "question": "什么是词性标注？常见的词性标签有哪些？"},
    {"qid": "Q8", "difficulty": "easy", "question": "什么是命名实体识别（NER）？它能识别哪些类型的实体？"},
    {"qid": "Q9", "difficulty": "easy", "question": "什么是词袋模型（Bag of Words）？"},
    {"qid": "Q10", "difficulty": "easy", "question": "什么是n-gram模型？它的优缺点是什么？"},
    {"qid": "Q11", "difficulty": "easy", "question": "什么是语言模型？它的主要作用是什么？"},
    {"qid": "Q12", "difficulty": "easy", "question": "什么是准确率（Precision）和召回率（Recall）？"},
    {"qid": "Q13", "difficulty": "easy", "question": "F1分数的计算公式是什么？它有什么意义？"},
    {"qid": "Q14", "difficulty": "easy", "question": "什么是过拟合？如何避免NLP模型过拟合？"},
    {"qid": "Q15", "difficulty": "easy", "question": "什么是交叉验证？在NLP任务中如何应用？"},

    # 二、传统方法与经典模型（16-30题，中级）
    {"qid": "Q16", "difficulty": "medium", "question": "隐马尔可夫模型（HMM）的三个基本问题是什么？"},
    {"qid": "Q17", "difficulty": "medium", "question": "最大熵模型的核心思想是什么？"},
    {"qid": "Q18", "difficulty": "medium", "question": "条件随机场（CRF）在序列标注任务中的优势是什么？"},
    {"qid": "Q19", "difficulty": "medium", "question": "什么是TF-IDF？它如何衡量词的重要性？"},
    {"qid": "Q20", "difficulty": "medium", "question": "决策树在文本分类任务中的工作原理是什么？"},
    {"qid": "Q21", "difficulty": "medium", "question": "支持向量机（SVM）为什么适合处理高维文本数据？"},
    {"qid": "Q22", "difficulty": "medium", "question": "什么是主题模型？LDA（潜在狄利克雷分配）的基本原理是什么？"},
    {"qid": "Q23", "difficulty": "medium", "question": "编辑距离（Levenshtein距离）如何计算？在NLP中有哪些应用？"},
    {"qid": "Q24", "difficulty": "medium",
     "question": "什么是词干提取（Stemming）和词形还原（Lemmatization）？两者的区别是什么？"},
    {"qid": "Q25", "difficulty": "medium", "question": "基于规则的NLP方法和基于统计的方法有何不同？"},
    {"qid": "Q26", "difficulty": "medium", "question": "如何用互信息（Mutual Information）衡量词语之间的关联性？"},
    {"qid": "Q27", "difficulty": "medium", "question": "什么是困惑度（Perplexity）？它如何评估语言模型的性能？"},
    {"qid": "Q28", "difficulty": "medium", "question": "传统机器翻译方法（如统计机器翻译）的局限性是什么？"},
    {"qid": "Q29", "difficulty": "medium", "question": "什么是情感分析？基于词典的情感分析方法原理是什么？"},
    {"qid": "Q30", "difficulty": "medium", "question": "什么是共指消解（Coreference Resolution）？举例说明其应用场景。"},

    # 三、深度学习与前沿技术（31-50题，复杂）
    {"qid": "Q31", "difficulty": "hard", "question": "神经网络如何处理文本数据？为什么需要词嵌入？"},
    {"qid": "Q32", "difficulty": "hard",
     "question": "词嵌入（Word Embedding）与独热编码（One-Hot Encoding）相比有什么优势？"},
    {"qid": "Q33", "difficulty": "hard", "question": "Word2Vec的CBOW和Skip-gram模型的核心区别是什么？"},
    {"qid": "Q34", "difficulty": "hard", "question": "GloVe词向量的训练原理是什么？它与Word2Vec有何不同？"},
    {"qid": "Q35", "difficulty": "hard", "question": "什么是上下文相关词向量？ELMo模型如何实现这一功能？"},
    {"qid": "Q36", "difficulty": "hard",
     "question": "CNN（卷积神经网络）在文本分类中的工作原理是什么？如何捕捉不同长度的语义特征？"},
    {"qid": "Q37", "difficulty": "hard",
     "question": "RNN（循环神经网络）为什么适合处理序列数据？它的梯度消失/爆炸问题如何解决？"},
    {"qid": "Q38", "difficulty": "hard",
     "question": "LSTM（长短期记忆网络）的门控机制（输入门、遗忘门、输出门）分别起什么作用？"},
    {"qid": "Q39", "difficulty": "hard", "question": "GRU（门控循环单元）与LSTM相比有哪些简化？在哪些场景下更适用？"},
    {"qid": "Q40", "difficulty": "hard", "question": "Transformer模型的核心创新点是什么？为什么能替代RNN成为主流架构？"},
    {"qid": "Q41", "difficulty": "hard", "question": "自注意力（Self-Attention）机制如何计算？它为什么能捕捉长距离依赖？"},
    {"qid": "Q42", "difficulty": "hard", "question": "多头注意力（Multi-Head Attention）的作用是什么？如何提升模型性能？"},
    {"qid": "Q43", "difficulty": "hard",
     "question": "BERT模型的预训练任务（Masked LM和Next Sentence Prediction）分别是什么？"},
    {"qid": "Q44", "difficulty": "hard", "question": "GPT模型的自回归生成方式与BERT的双向编码有何本质区别？"},
    {"qid": "Q45", "difficulty": "hard",
     "question": "什么是微调（Fine-tuning）？在预训练模型应用中如何平衡微调与泛化能力？"},
    {"qid": "Q46", "difficulty": "hard", "question": "什么是零样本学习（Zero-shot Learning）？在NLP任务中如何实现？"},
    {"qid": "Q47", "difficulty": "hard",
     "question": "大语言模型（如GPT-4、LLaMA）的涌现能力（Emergent Abilities）指什么？举例说明。"},
    {"qid": "Q48", "difficulty": "hard",
     "question": "什么是提示工程（Prompt Engineering）？如何通过提示词提升模型在特定任务上的表现？"},
    {"qid": "Q49", "difficulty": "hard",
     "question": "知识蒸馏（Knowledge Distillation）如何将大模型的能力迁移到小模型？在NLP中有何应用？"},
    {"qid": "Q50", "difficulty": "hard",
     "question": "联邦学习（Federated Learning）在NLP中的优势是什么？如何解决数据隐私与模型性能的矛盾？"},
]

# 2. 人工标注的“相关文档分块ID”（示例，对应报告中Precision@k计算）
# 实际使用时需根据文档分块的真实ID补充（可从metadata.jsonl中获取）
RELEVANT_CHUNKS: Dict[str, List[str]] = {
    "Q1": ["chunk_12", "chunk_15", "chunk_23"],  # “什么是NLP”的相关分块ID
    "Q18": ["chunk_102", "chunk_105", "chunk_110"],  # “CRF优势”的相关分块ID
    "Q40": ["chunk_301", "chunk_305", "chunk_312"],  # “Transformer创新点”的相关分块ID
    # 其他问题的相关分块ID需逐一补充
}


def batch_run_tests(output_path: str = "experiment_results.jsonl"):
    """
    批量运行测试集，记录结果并保存到JSONL文件
    """
    # 初始化聊天机器人（复用主程序逻辑）
    chatbot = TechDocChatbot()
    evaluator = RAGEvaluator()  # 用于实时计算部分指标

    # 初始化系统（加载模型、索引）
    if not chatbot.initialize_system():
        print("系统初始化失败，终止批量测试")
        return

    # 打开结果文件（追加模式，支持中断后继续）
    with open(output_path, "a", encoding="utf-8") as f:
        # 遍历所有测试问题
        for idx, test_q in enumerate(NLP_TEST_QUESTIONS, 1):
            qid = test_q["qid"]
            difficulty = test_q["difficulty"]
            question = test_q["question"]
            print(f"=== 正在测试 {qid}（{difficulty}）[{idx}/50]：{question} ===")

            try:
                # 记录开始时间（计算响应延迟）
                start_time = time.time()

                # 执行查询（复用主程序的process_query）
                answer, retrieved_chunks = chatbot.process_query(question)

                # 计算响应延迟（秒）
                response_time = round(time.time() - start_time, 2)

                # 计算核心指标（检索质量、回答质量的部分自动指标）
                retrieval_metrics = {}
                if qid in RELEVANT_CHUNKS:  # 若有标注的相关分块，计算检索指标
                    retrieval_metrics = evaluator.evaluate_retrieval(
                        retrieved_chunks=retrieved_chunks,
                        relevant_chunk_ids=RELEVANT_CHUNKS[qid],
                        top_k=5
                    )

                # 记录检索分块的关键信息（仅保留必要字段，避免冗余）
                retrieved_info = [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "source": chunk["source"],
                        "distance": round(chunk["distance"], 4),
                        "content_snippet": chunk["content"][:100] + "...",  # 保留片段（不影响其他逻辑）
                        "content": chunk["content"]  # 关键：补回完整content字段！
                    }
                    for chunk in retrieved_chunks[:5]  # 仅保留前5个检索结果
                ]

                # 组装当前问题的完整结果
                result = {
                    "qid": qid,
                    "difficulty": difficulty,
                    "question": question,
                    "answer": answer,
                    "response_time": response_time,
                    "retrieved_chunks": retrieved_info,
                    "retrieval_metrics": retrieval_metrics,
                    "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "success"
                }

                # 保存结果到JSONL
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()  # 实时写入，避免缓存丢失

                # 打印关键结果（便于实时观察）
                print(f" 测试成功 | 响应时间：{response_time}s | 检索分块数：{len(retrieved_chunks)}")
                print(f" 回答片段：{answer[:150]}..." if len(answer) > 150 else f"回答：{answer}")
                print("-" * 80 + "\n")

            except Exception as e:
                # 记录异常情况
                error_result = {
                    "qid": qid,
                    "difficulty": difficulty,
                    "question": question,
                    "error_message": str(e),
                    "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "failed"
                }
                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                f.flush()
                print(f" 测试失败：{str(e)}")
                print("-" * 80 + "\n")
                continue

    print(f"批量测试完成！所有结果已保存到：{os.path.abspath(output_path)}")


# 3. 人工评分录入代码（用于补充报告中的“用户评分”指标）
def add_manual_rating(input_path: str = "experiment_results.jsonl",
                      output_path: str = "experiment_results_with_rating.jsonl"):
    """
    读取批量测试结果，手动录入用户评分（1-5星），生成带评分的结果文件
    """
    with open(input_path, "r", encoding="utf-8") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            result = json.loads(line.strip())
            if result["status"] != "success":
                out_f.write(line)
                continue

            # 显示问题和回答，提示录入评分
            print(f"\n=== {result['qid']}（{result['difficulty']}）===")
            print(f"问题：{result['question']}")
            print(f"回答：{result['answer']}")

            # 循环获取有效评分（1-5）
            while True:
                rating_input = input("请输入评分（1-5星）：")
                if rating_input.isdigit() and 1 <= int(rating_input) <= 5:
                    result["user_rating"] = int(rating_input)
                    break
                print("无效评分！请输入1-5之间的整数。")

            # 保存带评分的结果
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"评分录入完成！结果保存到：{os.path.abspath(output_path)}")


if __name__ == "__main__":
    # 第一步：批量运行测试（生成基础结果）
    batch_run_tests()
    # 第二步：手动录入评分（关键！生成带评分的文件）
    add_manual_rating()