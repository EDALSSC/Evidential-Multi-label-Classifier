# llm_clients.py
import os
from openai import OpenAI
from zhipuai import ZhipuAI

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

def call_qwen(prompt: str, timeout: int = 30) -> str:
    if not DASHSCOPE_API_KEY:
        return "❌ 未配置 DASHSCOPE_API_KEY"
    try:
        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Qwen 调用失败: {str(e)}"

def call_glm(prompt: str, timeout: int = 30) -> str:
    if not ZHIPUAI_API_KEY:
        return "❌ 未配置 ZHIPUAI_API_KEY"
    try:
        client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ GLM 调用失败: {str(e)}"
    

    # llm_clients.py 新增
def call_deepseek(prompt: str, timeout: int = 30) -> str:
    
    if not DEEPSEEK_API_KEY:
        return "❌ 未配置 DEEPSEEK_API_KEY"
    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ DeepSeek 调用失败: {str(e)}"
    

# llm_clients.py —— 新增 Moonshot 支持

def call_moonshot(prompt: str, timeout: int = 30) -> str:
    if not MOONSHOT_API_KEY:
        return "❌ 未配置 MOONSHOT_API_KEY"
    try:
        client = OpenAI(
            api_key=MOONSHOT_API_KEY,
            base_url="https://api.moonshot.cn/v1"
        )
        response = client.chat.completions.create(
            model="moonshot-v1-8k",  # 也可用 moonshot-v1-32k / v1-128k
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "Insufficient Balance" in error_msg:
            return "💰 Moonshot 余额不足，请登录 https://www.moonshot.cn 充值"
        else:
            return f"❌ Moonshot 调用失败: {error_msg}"