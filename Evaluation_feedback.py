import json
import os
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
from sklearn.metrics import precision_score, recall_score


class RAGEvaluator:
    """RAG系统评估器，支持检索质量和回答质量评估"""

    def __init__(self):
        # 初始化评估工具
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.bleu_smoother = SmoothingFunction().method4  # BLEU平滑函数

    def evaluate_retrieval(
            self,
            retrieved_chunks: List[Dict],
            relevant_chunk_ids: List[str],
            top_k: int = 5
    ) -> Dict[str, float]:
        """
        评估检索质量：计算准确率、召回率、MRR（平均倒数排名）

        Args:
            retrieved_chunks: 检索到的分块列表（含chunk_id）
            relevant_chunk_ids: 人工标注的相关分块ID列表
            top_k: 评估前k个检索结果

        Returns:
            检索评估指标字典
        """
        # 提取前k个检索结果的ID
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks[:top_k]]
        total_relevant = len(relevant_chunk_ids)
        retrieved_count = len(retrieved_ids)

        # 1. 准确率（P@k）：前k个结果中相关分块的比例
        relevant_in_retrieved = [1 if idx in relevant_chunk_ids else 0 for idx in retrieved_ids]
        precision = sum(relevant_in_retrieved) / retrieved_count if retrieved_count > 0 else 0.0

        # 2. 召回率（R@k）：检索到的相关分块占所有相关分块的比例
        recall = sum(relevant_in_retrieved) / total_relevant if total_relevant > 0 else 0.0

        # 3. MRR（平均倒数排名）：第一个相关分块的倒数排名
        mrr = 0.0
        for rank, idx in enumerate(retrieved_ids, 1):
            if idx in relevant_chunk_ids:
                mrr = 1.0 / rank
                break

        return {
            "retrieval_precision": round(precision, 4),
            "retrieval_recall": round(recall, 4),
            "retrieval_mrr": round(mrr, 4),
            "top_k": top_k
        }

    def check_source_consistency(
            self,
            answer: str,
            retrieved_chunks: List[Dict]
    ) -> float:
        """
        检查回答与检索来源的一致性：回答中的关键信息是否可在检索分块中找到

        Returns:
            一致性分数（0-1，1表示完全一致）
        """
        if not answer or not retrieved_chunks:
            return 0.0

        # 提取回答中的关键术语（排除停用词）
        answer_terms = self._extract_key_terms(answer)
        if not answer_terms:
            return 0.0

        # 提取所有检索分块的文本
        chunk_texts = " ".join([chunk["content"] for chunk in retrieved_chunks])

        # 计算关键术语的覆盖率
        covered_terms = [term for term in answer_terms if term in chunk_texts]
        return round(len(covered_terms) / len(answer_terms), 4)

    def evaluate_answer(
            self,
            answer: str,
            reference_answer: str,
            retrieved_chunks: List[Dict]
    ) -> Dict[str, float]:
        """
        评估回答质量：计算ROUGE-L（与参考回答重叠度）、BLEU（流畅度）、来源一致性

        Args:
            answer: 模型生成的回答
            reference_answer: 人工标注的参考回答
            retrieved_chunks: 检索到的分块列表

        Returns:
            回答评估指标字典
        """
        # 1. ROUGE-L（衡量与参考回答的语义重叠）
        rouge_score = self.rouge_scorer.score(reference_answer, answer)["rougeL"].fmeasure

        # 2. BLEU（衡量生成文本的流畅度和准确性）
        answer_tokens = answer.lower().split()
        reference_tokens = [reference_answer.lower().split()]
        bleu_score = sentence_bleu(
            reference_tokens,
            answer_tokens,
            smoothing_function=self.bleu_smoother
        )

        # 3. 来源一致性
        source_consistency = self.check_source_consistency(answer, retrieved_chunks)

        return {
            "answer_rougeL": round(rouge_score, 4),
            "answer_bleu": round(bleu_score, 4),
            "source_consistency": source_consistency
        }

    def _extract_key_terms(self, text: str) -> List[str]:
        """提取文本中的关键术语（过滤停用词和短词）"""
        stop_words = {"的", "了", "在", "是", "和", "有", "为", "与", "之", "等", "the", "a", "an", "is", "are"}
        # 提取单词和专业术语（字母、数字、下划线）
        terms = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
        # 过滤短词（长度<2）和停用词
        return [term for term in terms if len(term) >= 2 and term not in stop_words]


class FeedbackCollector:
    """用户反馈收集器，支持存储和查询反馈数据"""

    def __init__(self, feedback_path: str = "rag_feedback.jsonl"):
        self.feedback_path = feedback_path
        # 确保反馈文件存在
        if not os.path.exists(feedback_path):
            with open(feedback_path, "w", encoding="utf-8") as f:
                pass  # 创建空文件

    def collect_feedback(
            self,
            query: str,
            answer: str,
            retrieved_chunks: List[Dict],
            user_rating: int,  # 1-5分，5分最好
            user_correction: Optional[str] = None,
            evaluation_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        收集用户反馈并存储到文件

        Args:
            query: 用户查询
            answer: 模型生成的回答
            retrieved_chunks: 检索到的分块
            user_rating: 用户对回答的评分（1-5）
            user_correction: 用户提供的修正内容（可选）
            evaluation_metrics: 自动评估指标（可选）

        Returns:
            反馈数据字典
        """
        feedback = {
            "feedback_id": f"fb_{int(time.time() * 1000)}",  # 唯一ID
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "answer": answer,
            "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
            "user_rating": user_rating,
            "user_correction": user_correction,
            "auto_evaluation": evaluation_metrics
        }

        # 追加到反馈文件（JSONL格式，每行一个反馈）
        with open(self.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")

        print(f"反馈已保存（ID: {feedback['feedback_id']}）")
        return feedback

    def load_feedback(self) -> List[Dict]:
        """加载历史反馈数据"""
        feedbacks = []
        with open(self.feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                feedbacks.append(json.loads(line.strip()))
        return feedbacks

    def analyze_feedback(self) -> Dict:
        """分析历史反馈，生成统计信息"""
        feedbacks = self.load_feedback()
        if not feedbacks:
            return {"message": "无反馈数据"}

        # 计算平均评分
        avg_rating = np.mean([fb["user_rating"] for fb in feedbacks])

        # 统计低评分（≤2分）的问题类型
        low_rating_queries = [fb["query"] for fb in feedbacks if fb["user_rating"] <= 2]
        low_rating_terms = [term for fb in low_rating_queries for term in RAGEvaluator()._extract_key_terms(fb)]
        top_low_terms = {term: low_rating_terms.count(term) for term in set(low_rating_terms)}
        top_low_terms = sorted(top_low_terms.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_feedback": len(feedbacks),
            "avg_user_rating": round(avg_rating, 2),
            "low_rating_count": len(low_rating_queries),
            "top_low_rating_terms": top_low_terms  # 低评分查询中高频出现的术语
        }


# 测试代码
if __name__ == "__main__":
    # 1. 初始化评估器和反馈收集器
    evaluator = RAGEvaluator()
    feedback_collector = FeedbackCollector()

    # 2. 模拟数据（实际使用时替换为真实系统输出）
    # 模拟检索结果
    retrieved_chunks = [
        {"chunk_id": "ch5_001", "content": "Word2vec包含CBOW和Skip-gram两种模型...", "source": "ch5.pdf"},
        {"chunk_id": "ch5_003", "content": "CBOW通过上下文预测中心词，Skip-gram相反...", "source": "ch5.pdf"},
        {"chunk_id": "ch6_002", "content": "Transformer使用自注意力机制...", "source": "ch6.pdf"}
    ]
    # 人工标注的相关分块ID
    relevant_chunk_ids = ["ch5_001", "ch5_003"]
    # 模型生成的回答
    generated_answer = "Word2vec包含两种模型：CBOW和Skip-gram。CBOW通过上下文预测中心词，Skip-gram通过中心词预测上下文【来源：ch5.pdf】。"
    # 参考回答（人工标注）
    reference_answer = "Word2vec的CBOW模型通过上下文词向量预测中心词，而Skip-gram模型通过中心词预测上下文词，两者均用于生成词向量。"
    # 用户查询
    test_query = "Word2vec的CBOW和Skip-gram有什么区别？"

    # 3. 评估检索质量
    retrieval_metrics = evaluator.evaluate_retrieval(
        retrieved_chunks=retrieved_chunks,
        relevant_chunk_ids=relevant_chunk_ids,
        top_k=3
    )
    print("检索评估结果：")
    print(retrieval_metrics)

    # 4. 评估回答质量
    answer_metrics = evaluator.evaluate_answer(
        answer=generated_answer,
        reference_answer=reference_answer,
        retrieved_chunks=retrieved_chunks
    )
    print("\n回答评估结果：")
    print(answer_metrics)

    # 5. 收集用户反馈（模拟用户评分4分，无修正）
    feedback = feedback_collector.collect_feedback(
        query=test_query,
        answer=generated_answer,
        retrieved_chunks=retrieved_chunks,
        user_rating=4,
        evaluation_metrics={**retrieval_metrics, **answer_metrics}
    )

    # 6. 分析反馈
    feedback_analysis = feedback_collector.analyze_feedback()
    print("\n反馈分析：")
    print(feedback_analysis)