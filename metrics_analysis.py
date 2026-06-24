import json
import numpy as np
from typing import List, Dict
from Evaluation_feedback import RAGEvaluator
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 1. 加载人工参考回答（与测试问题对应，用于计算ROUGE-L、BLEU）
# 实际使用时需根据NLP权威资料补充完整参考回答
REFERENCE_ANSWERS: Dict[str, str] = {
    "Q1": "自然语言处理（NLP）是人工智能的一个分支，研究计算机理解、分析、生成人类语言的技术，核心目标是实现人机语言交互。",
    "Q18": "条件随机场（CRF）在序列标注任务中具有全局最优性，能利用上下文信息建模标签依赖关系，避免局部最优；同时支持自定义特征函数，适配NLP任务（如NER、词性标注）的领域知识。",
    "Q40": "Transformer的核心创新点包括自注意力机制（并行计算、捕捉长距离依赖）、位置编码（补充时序信息）、编码器-解码器架构（适配生成任务）；替代RNN的原因是解决了RNN的序列依赖（无法并行）和长距离依赖丢失问题，训练效率与模型性能更优。",
    # 其他问题的参考回答需逐一补充
}


def load_experiment_results(result_path: str = "experiment_results_with_rating.jsonl") -> List[Dict]:
    """加载实验结果文件"""
    results = []
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return [r for r in results if r["status"] == "success"]  # 仅保留成功的结果


def calculate_overall_metrics(success_results: List[Dict]) -> Dict:
    """
    计算整体性能指标，对应报告中的“实验结果与分析”部分
    """
    evaluator = RAGEvaluator()
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_smoother = SmoothingFunction().method4

    # 按难度分组统计
    metrics_by_difficulty = {
        "easy": {"results": [], "retrieval": [], "answer": [], "rating": []},
        "medium": {"results": [], "retrieval": [], "answer": [], "rating": []},
        "hard": {"results": [], "retrieval": [], "answer": [], "rating": []}
    }

    # 遍历所有成功结果，按难度分组
    for res in success_results:
        difficulty = res["difficulty"]
        metrics_by_difficulty[difficulty]["results"].append(res)

        # 1. 收集检索指标（仅当有retrieval_metrics时）
        if res.get("retrieval_metrics"):
            metrics_by_difficulty[difficulty]["retrieval"].append(res["retrieval_metrics"])

        # 2. 收集用户评分
        if "user_rating" in res:
            metrics_by_difficulty[difficulty]["rating"].append(res["user_rating"])

        # 3. 计算回答质量指标（ROUGE-L、BLEU）
        if res["qid"] in REFERENCE_ANSWERS:
            ref_answer = REFERENCE_ANSWERS[res["qid"]]
            gen_answer = res["answer"]

            # 计算ROUGE-L
            rouge_score = rouge_scorer_obj.score(ref_answer, gen_answer)["rougeL"].fmeasure

            # 计算BLEU
            gen_tokens = gen_answer.lower().split()
            ref_tokens = [ref_answer.lower().split()]
            bleu_score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=bleu_smoother)

            metrics_by_difficulty[difficulty]["answer"].append({
                "rougeL": rouge_score,
                "bleu": bleu_score,
                "source_consistency": evaluator.check_source_consistency(gen_answer, res["retrieved_chunks"])
            })

    # 计算各难度的平均指标
    overall_metrics = {"by_difficulty": {}, "total": {}}
    total_retrieval = []
    total_answer = []
    total_rating = []

    for difficulty, data in metrics_by_difficulty.items():
        # 检索指标平均
        avg_retrieval = {}
        if data["retrieval"]:
            avg_retrieval = {
                "precision@5": round(np.mean([m["retrieval_precision"] for m in data["retrieval"]]), 4),
                "mrr": round(np.mean([m["retrieval_mrr"] for m in data["retrieval"]]), 4),
                "count": len(data["retrieval"])
            }

        # 回答指标平均
        avg_answer = {}
        if data["answer"]:
            avg_answer = {
                "rougeL": round(np.mean([m["rougeL"] for m in data["answer"]]), 4),
                "bleu": round(np.mean([m["bleu"] for m in data["answer"]]), 4),
                "source_consistency": round(np.mean([m["source_consistency"] for m in data["answer"]]), 4),
                "count": len(data["answer"])
            }

        # 用户评分平均
        avg_rating = {}
        if data["rating"]:
            avg_rating = {
                "avg_score": round(np.mean(data["rating"]), 2),
                "distribution": {str(star): data["rating"].count(star) for star in range(1, 6)},
                "count": len(data["rating"])
            }

        # 响应延迟平均
        avg_response_time = round(np.mean([res["response_time"] for res in data["results"]]), 2)

        # 检索覆盖率（有检索结果的问题占比）
        retrieval_coverage = round(
            len([res for res in data["results"] if len(res["retrieved_chunks"]) > 0]) / len(data["results"]), 4)

        # 保存当前难度的指标
        overall_metrics["by_difficulty"][difficulty] = {
            "sample_count": len(data["results"]),
            "retrieval_coverage": retrieval_coverage,
            "avg_response_time": avg_response_time,
            "retrieval_metrics": avg_retrieval,
            "answer_metrics": avg_answer,
            "user_rating": avg_rating
        }

        # 汇总到整体指标
        total_retrieval.extend(data["retrieval"])
        total_answer.extend(data["answer"])
        total_rating.extend(data["rating"])

    # 计算整体平均指标
    overall_metrics["total"] = {
        "sample_count": sum([len(data["results"]) for data in metrics_by_difficulty.values()]),
        "avg_response_time": round(
            np.mean([res["response_time"] for data in metrics_by_difficulty.values() for res in data["results"]]), 2),
        "retrieval_coverage": round(
            sum([len([res for res in data["results"] if len(res["retrieved_chunks"]) > 0]) for data in
                 metrics_by_difficulty.values()]) /
            sum([len(data["results"]) for data in metrics_by_difficulty.values()]),
            4
        ),
        "retrieval_metrics": {
            "precision@5": round(np.mean([m["retrieval_precision"] for m in total_retrieval]),
                                 4) if total_retrieval else {},
            "mrr": round(np.mean([m["retrieval_mrr"] for m in total_retrieval]), 4) if total_retrieval else {},
            "count": len(total_retrieval)
        },
        "answer_metrics": {
            "rougeL": round(np.mean([m["rougeL"] for m in total_answer]), 4) if total_answer else {},
            "bleu": round(np.mean([m["bleu"] for m in total_answer]), 4) if total_answer else {},
            "source_consistency": round(np.mean([m["source_consistency"] for m in total_answer]),
                                        4) if total_answer else {},
            "count": len(total_answer)
        },
        "user_rating": {
            "avg_score": round(np.mean(total_rating), 2) if total_rating else {},
            "count": len(total_rating)
        }
    }

    return overall_metrics


def print_metrics_report(overall_metrics: Dict):
    """打印性能指标报告（格式对齐，便于直接复制到报告中）"""
    print("=" * 100)
    print("                        技术文档问答系统性能指标报告")
    print("=" * 100)

    # 打印整体指标
    total = overall_metrics["total"]
    print(f"\n【整体指标】")
    print(
        f"样本总数：{total['sample_count']} | 平均响应延迟：{total['avg_response_time']}s | 检索覆盖率：{total['retrieval_coverage'] * 100:.2f}%")
    print(
        f"检索质量：Precision@5={total['retrieval_metrics'].get('precision@5', 0) * 100:.2f}% | MRR={total['retrieval_metrics'].get('mrr', 0):.4f}")
    print(
        f"回答质量：ROUGE-L={total['answer_metrics'].get('rougeL', 0):.4f} | BLEU={total['answer_metrics'].get('bleu', 0):.4f} | 来源一致性={total['answer_metrics'].get('source_consistency', 0) * 100:.2f}%")
    print(f"用户评分：平均{total['user_rating'].get('avg_score', 0)}星（{total['user_rating'].get('count', 0)}个样本）")

    # 按难度打印详细指标
    for difficulty, metrics in overall_metrics["by_difficulty"].items():
        diff_cn = {"easy": "简单", "medium": "中级", "hard": "复杂"}[difficulty]
        print(f"\n【{diff_cn}问题（{metrics['sample_count']}个）】")
        print(f"检索覆盖率：{metrics['retrieval_coverage'] * 100:.2f}% | 平均响应延迟：{metrics['avg_response_time']}s")

        if metrics["retrieval_metrics"]:
            print(
                f"检索指标：Precision@5={metrics['retrieval_metrics']['precision@5'] * 100:.2f}% | MRR={metrics['retrieval_metrics']['mrr']:.4f}")

        if metrics["answer_metrics"]:
            print(
                f"回答指标：ROUGE-L={metrics['answer_metrics']['rougeL']:.4f} | BLEU={metrics['answer_metrics']['bleu']:.4f} | 来源一致性={metrics['answer_metrics']['source_consistency'] * 100:.2f}%")

        if metrics["user_rating"]:
            print(
                f"用户评分：平均{metrics['user_rating']['avg_score']}星 | 分布：{metrics['user_rating']['distribution']}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    # 加载实验结果（需先运行batch_test.py并录入评分）
    success_results = load_experiment_results()
    print(f"加载到{len(success_results)}个有效实验结果")

    # 计算整体指标
    overall_metrics = calculate_overall_metrics(success_results)

    # 打印报告（可直接复制到课程设计报告中）
    print_metrics_report(overall_metrics)

    # 保存指标结果到JSON（便于后续可视化）
    with open("experiment_overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n指标结果已保存到：experiment_overall_metrics.json")