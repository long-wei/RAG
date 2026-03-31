import re
from typing import List, Dict, Optional
import os


def build_tech_prompt(
        query: str,
        retrieved_chunks: List[Dict],
        max_context_length: int = 2000  # 控制上下文总长度，避免超过LLM限制
) -> str:
    """
    构建技术文档专用提示：
    1. 明确指令：基于上下文回答，标注来源，禁止编造
    2. 结构化上下文：按相关性排序，附来源信息
    3. 长度控制：超过max_context_length时截断相关性较低的分块
    长度控制注：
    1. 智能排序：优先保留与查询最相关的内容
    2. 分级截断：先截断长内容，再移除低优先级分块
    3. 摘要机制：对超长内容生成摘要
    """
    # 1. 定义系统指令（核心约束）
    system_prompt = """
    你是技术文档问答专家，专门负责基于提供的技术文档内容准确回答用户问题：
    - 回答必须准确反映文档内容
    - 当文档内容不足以回答问题时，可参考权威学术资源进行补充整合，但需明确标注信息来源
    - 给出的回答必须准确且流畅（用户无明确详细回答时内容需简洁明了）
    - 具备对话记忆能力，能够结合历史对话内容理解并回答后续问题
    - 在涉及政治、社会、个人等的敏感问题时拒绝回答
    - 关键结论需标注来源（格式：【来源：xxx】）
    - 公式用$包裹（如：__E=mc^2$__），代码块用--标注（如：--print("Hello")--)
    """

    # 2. 智能组装上下文
    context_parts = []
    total_length = 0

    # 按相关性排序（假设chunks已按相关性排序）
    for i, chunk in enumerate(retrieved_chunks):
        # 单个上下文片段格式："[来源：xxx] 内容：xxx"
        part = f"[来源：{chunk['source']}] 内容：{chunk['content']}"
        part_length = len(part)

        # 如果单个分块就超过长度限制，考虑摘要或截断
        if part_length > max_context_length * 0.5:  # 单个分块不超过总长度的一半
            # 可以考虑提取关键句子或生成摘要
            truncated_content = chunk['content'][:int(max_context_length * 0.4)]
            part = f"[来源：{chunk['source']}] 内容：{truncated_content}..."
            part_length = len(part)

        # 控制总长度
        if total_length + part_length > max_context_length:
            # 可以添加提示说明内容被截断
            context_parts.append("[注意：由于长度限制，部分内容已被省略]")
            break

        context_parts.append(part)
        total_length += part_length

    # 3. 拼接完整提示
    context = "\n\n".join(context_parts) if context_parts else "无相关技术文档内容"
    final_prompt = f"{system_prompt}\n\n技术文档上下文：\n{context}\n\n用户问题：{query}\n\n回答："

    return final_prompt.strip()


def load_llm(api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1") -> dict:
    """
    初始化阿里云百炼API客户端配置
    
    Args:
        api_key: 阿里云百炼API Key，如果为None则从环境变量获取
        base_url: API基础URL
    
    Returns:
        dict: API配置信息
    """
    import os
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not api_key:
        raise ValueError("未提供API Key，请设置DASHSCOPE_API_KEY环境变量或在调用时传入api_key参数")
    
    return {
        "api_key": api_key,
        "base_url": base_url
    }


def generate_tech_answer(
        prompt: str,
        api_config: dict,
        max_tokens: int = 512,
        model: str = "qwen-plus"
) -> str:
    """
    通过阿里云百炼API生成回答
    
    Args:
        prompt: 构建好的提示词
        api_config: API配置信息，由load_llm函数返回
        max_tokens: 最大生成token数
        model: 使用的模型名称
    
    Returns:
        str: 生成的回答
    """
    from openai import OpenAI
    
    client = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"],
    )
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': '你是一个技术文档问答专家，请严格基于提供的技术文档内容回答问题。'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"调用阿里云百炼API时出错: {str(e)}")


# 测试代码
if __name__ == "__main__":
    # 加载依赖组件
    from index_loader import load_all_components
    from query_processor import generate_query_embedding
    from retrieval_optimization import retrieve_and_optimize

    # 1. 加载基础组件
    tokenizer, vector_model, index, metadata = load_all_components()
    # 加载API配置
    llm_config = load_llm()  # 确保已设置DASHSCOPE_API_KEY环境变量

    # 2. 示例查询流程
    test_query = "对比Word2vec的CBOW和Skip-gram模型的原理及应用场景"
    query_emb = generate_query_embedding(test_query, tokenizer, vector_model)

    # 3. 检索优化
    retrieved_chunks = retrieve_and_optimize(
        query_embedding=query_emb,
        index=index,
        metadata=metadata,
        query_text=test_query
    )

    # 4. 构建提示
    prompt = build_tech_prompt(test_query, retrieved_chunks)
    print(f"构建的提示（前500字符）：\n{prompt[:500]}...")

    # 5. 生成回答
    answer = generate_tech_answer(prompt, llm_config)
    print(f"\n生成的回答：\n{answer}")