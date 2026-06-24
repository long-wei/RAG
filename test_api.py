"""
测试阿里云百炼API集成
"""
import os
from LLM_Integration import load_llm, generate_tech_answer


def test_api_integration():
    """测试API集成"""
    try:
        # 加载API配置
        print("正在加载阿里云百炼API配置...")
        api_config = load_llm()
        print("API配置加载成功")

        # 构建测试提示
        test_prompt = """
        你是技术文档问答专家，需严格基于以下提供的技术文档内容回答问题：
        - 回答必须准确反映文档内容并结合权威学术网站的相关资料内容进行整合，确保给出的回答准确且流畅
        - 若文档中无相关信息，直接回复"未找到与该问题相关的技术内容"

        技术文档上下文：
        无相关技术文档内容

        用户问题：你好，请简单介绍一下 yourself

        回答：
        """

        # 生成回答
        print("正在调用阿里云百炼API生成回答...")
        answer = generate_tech_answer(test_prompt, api_config, max_tokens=200)
        print("API调用成功！")
        print(f"生成的回答：\n{answer}")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")


if __name__ == "__main__":
    test_api_integration()
