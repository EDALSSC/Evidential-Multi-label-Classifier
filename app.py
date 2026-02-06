from dotenv import load_dotenv
import os
import gradio as gr
from typing import List, Dict, Any

load_dotenv()

from inference import predict
from llm_clients import call_qwen, call_glm, call_deepseek, call_moonshot

print("DEBUG: ZHIPUAI_API_KEY =", repr(os.getenv("ZHIPUAI_API_KEY")))
print("DEBUG: DASHSCOPE_API_KEY =", repr(os.getenv("DASHSCOPE_API_KEY")))
print("DEBUG: DEEPSEEK_API_KEY =", repr(os.getenv("DEEPSEEK_API_KEY")))
print("DEBUG: MOONSHOT_API_KEY =", repr(os.getenv("MOONSHOT_API_KEY")))

# 自动检测可用模型
AVAILABLE_MODELS = ["本地农业分类模型"]
if os.getenv("DASHSCOPE_API_KEY"):
    AVAILABLE_MODELS.append("Qwen 大模型")
if os.getenv("ZHIPUAI_API_KEY"):
    AVAILABLE_MODELS.append("GLM 大模型")
if os.getenv("DEEPSEEK_API_KEY"):
    AVAILABLE_MODELS.append("DeepSeek 大模型")
if os.getenv("MOONSHOT_API_KEY"):
    AVAILABLE_MODELS.append("Moonshot 大模型")

# 定义标签到模型的映射（支持多标签）
LABEL_TO_MODEL = {
    "原因类": "GLM 大模型",
    "定义类": "GLM 大模型", 
    "建议类": "Qwen 大模型",
    "查询类": "Moonshot 大模型",  # Moonshot 擅长解释和查询
    "结果类": "GLM 大模型",
    "解决类": "Qwen 大模型"       # Qwen 擅长解决方案
}

# 定义模型专长描述
MODEL_EXPERTISE = {
    "Qwen 大模型": "擅长提供解决方案和建议",
    "GLM 大模型": "擅长分析原因和定义概念",
    "Moonshot 大模型": "擅长解释查询类问题",
    "DeepSeek 大模型": "擅长深度分析"
}

def integrate_answers(question: str, individual_answers: dict, labels: list, model_usage_info: dict) -> str:
    """
    使用 Moonshot 将多个模型的回答整合成一个统一、美观、专业的农业专家级回答
    """
    if not os.getenv("MOONSHOT_API_KEY"):
        return f"""
        <div style="background:#fff9c4; border-left:4px solid #ffc107; padding:12px; border-radius:6px; margin:10px 0;">
            ❌ 无法整合答案：缺少 Moonshot API Key（整合功能必需）
        </div>
        """

    # 构建模型调用摘要（简洁版）
    model_calls_desc = []
    for label, (model_name, expertise) in model_usage_info.items():
        model_calls_desc.append(f"<span style='background:#e8f5e8; padding:2px 6px; border-radius:4px; font-size:0.85em;'>{label}</span> → {model_name}（{expertise}）")
    models_used_html = " | ".join(model_calls_desc)

    # 构建整合提示（保持原逻辑，但优化输出结构）
    answers_text = ""
    for label, answer in individual_answers.items():
        # 尝试提取关键段落（避免冗余）
        clean_ans = answer.strip()
        if clean_ans.startswith("【") and "】" in clean_ans:
            clean_ans = clean_ans.split("】", 1)[-1].strip()
        answers_text += f"<div class='answer-section' style='margin-bottom:16px;'><strong>📌 {label}视角：</strong><br>{clean_ans}</div>\n"

    integration_prompt = f"""
    原始问题：{question}

    请将以下从不同专业角度的回答，整合为一份面向农业技术人员的结构化报告，要求：
    1. 分模块组织：【问题概述】→【症状识别】→【发生原因】→【影响分析】→【防治建议】→【观察要点】
    2. 每个模块使用清晰标题（H3级），内容精炼，避免重复
    3. 关键术语加粗，重要操作步骤用✅符号标注
    4. 语言专业但易懂
    5. 输出纯 HTML 片段（不要 markdown，不要额外说明）

    回答来源：
    {answers_text}

    请直接输出整合后的 HTML 内容（仅内容区域，不包含 <html><body>）：
    """

    try:
        integrated_response = call_moonshot(integration_prompt).strip()

        # 如果返回的是 Markdown，尝试转换为简单 HTML（兜底）
        if integrated_response.startswith("#") or "**" in integrated_response:
            # 简单转换：标题→<h3>，**bold**→<strong>，- → ✅
            html_content = integrated_response
            html_content = html_content.replace("### ", "<h3 style='color:#2e7d32; margin:16px 0 8px 0;'>").replace("\n###", "</h3><h3 style='color:#2e7d32; margin:16px 0 8px 0;'>")
            html_content = html_content.replace("## ", "<h3 style='color:#2e7d32; margin:16px 0 8px 0;'>").replace("\n##", "</h3><h3 style='color:#2e7d32; margin:16px 0 8px 0;'>")
            html_content = html_content.replace("**", "<strong>").replace("**", "</strong>")
            html_content = html_content.replace("- ", "✅ ").replace("• ", "✅ ")
            html_content = html_content.replace("\n", "<br>")
            integrated_response = html_content

        # 最终封装为美观卡片
        result_html = f"""
        <div style="background:#ffffff; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.05); overflow:hidden; margin:12px 0;">
            <div style="background:linear-gradient(135deg, #2e7d32, #1b5e20); color:white; padding:14px 20px; font-weight:bold; display:flex; align-items:center; gap:8px;">
                🌾 【智能整合报告】—— 基于 {', '.join(labels)} 的多模协同分析
            </div>
            <div style="padding:20px; line-height:1.6; color:#333; font-size:14px;">
                <div style="font-size:0.9em; color:#666; margin-bottom:16px; padding-bottom:12px; border-bottom:1px dashed #eee;">
                    🔍 模型协作路径：{models_used_html}
                </div>

                {integrated_response}

                <div style="margin-top:24px; padding-top:16px; border-top:1px dashed #eee; font-size:0.85em; color:#777;">
                    💡 提示：本报告由多模型协同生成，适用于田间诊断与技术指导。实际应用请结合当地气候与品种调整。
                </div>
            </div>
        </div>
        """
        return result_html

    except Exception as e:
        # 兜底：即使失败也尽量美化
        model_calls_html = "<br>".join([
            f"🔹 {label} → {model_name}（{expertise}）"
            for label, (model_name, expertise) in model_usage_info.items()
        ])
        raw_answers_html = "".join([
            f"<div style='margin:8px 0; padding:10px; background:#f8f9fa; border-left:3px solid #4caf50;'><strong>{label}:</strong><br>{answer}</div>"
            for label, answer in individual_answers.items()
        ])

        return f"""
        <div style="background:#fff8e1; border-left:4px solid #ffa726; padding:16px; border-radius:8px; margin:12px 0;">
            <h3 style="color:#e65100; margin-top:0;">⚠️ 整合失败｜回退至原始模型回答</h3>
            <p><strong>错误：</strong>{str(e)}</p>
            <div style="margin-top:12px;">
                <strong>调用模型：</strong><br>{model_calls_html}
            </div>
            <div style="margin-top:16px;">
                <strong>原始回答：</strong><br>{raw_answers_html}
            </div>
        </div>
        """

def get_combined_answer(question: str, labels: list) -> str:
    """
    根据多个标签，调用不同模型，然后整合回答
    """
    individual_answers = {}
    unavailable_models = []
    model_usage_info = {}  # 记录模型使用信息
    
    # 为每个标签找到对应的模型并调用
    for label in labels:
        target_model = LABEL_TO_MODEL.get(label, "Moonshot 大模型")  # 默认改为Moonshot
        
        # 调用对应模型
        if target_model == "Qwen 大模型" and os.getenv("DASHSCOPE_API_KEY"):
            individual_answers[label] = call_qwen(f"关于问题：'{question}'，请从{label}的角度详细回答：")
            model_usage_info[label] = (target_model, MODEL_EXPERTISE[target_model])
        elif target_model == "GLM 大模型" and os.getenv("ZHIPUAI_API_KEY"):
            individual_answers[label] = call_glm(f"关于问题：'{question}'，请从{label}的角度详细回答：")
            model_usage_info[label] = (target_model, MODEL_EXPERTISE[target_model])
        elif target_model == "DeepSeek 大模型" and os.getenv("DEEPSEEK_API_KEY"):
            individual_answers[label] = call_deepseek(f"关于问题：'{question}'，请从{label}的角度详细回答：")
            model_usage_info[label] = (target_model, MODEL_EXPERTISE[target_model])
        elif target_model == "Moonshot 大模型" and os.getenv("MOONSHOT_API_KEY"):
            individual_answers[label] = call_moonshot(f"关于问题：'{question}'，请从{label}的角度详细回答：")
            model_usage_info[label] = (target_model, MODEL_EXPERTISE[target_model])
        else:
            # 如果目标模型不可用，记录下来
            unavailable_models.append(f"{label}({target_model})")
    
    # 如果有回答，进行整合
    if individual_answers:
        # 调用整合函数
        integrated_result = integrate_answers(question, individual_answers, labels, model_usage_info)
        
        # 添加不可用模型的提示
        if unavailable_models:
            integrated_result += f"\n\n<div style='background:#ffebee; border-left:4px solid #f44336; padding:10px; border-radius:6px; margin:10px 0; font-size:0.9em;'>⚠️ <strong>以下模型不可用</strong>：{', '.join(unavailable_models)}<br>请配置相应的 API Key。</div>"
        
        return integrated_result
    else:
        # 所有模型都不可用
        if unavailable_models:
            return f"""<div style="background:#ffebee; border-left:4px solid #f44336; padding:16px; border-radius:8px; margin:12px 0;">
                <h3 style="color:#c62828; margin-top:0;">❌ 【智能路由】所有目标模型均不可用</h3>
                <p>{', '.join(unavailable_models)}</p>
                <p>请配置相应的 API Key。</p>
            </div>"""
        else:
            return """<div style="background:#e3f2fd; border-left:4px solid #2196f3; padding:16px; border-radius:8px; margin:12px 0;">
                <h3 style="color:#1565c0; margin-top:0;">❌ 【智能路由】未能获取任何模型的回答</h3>
                <p>可能的原因是本地模型未匹配到任何预设类别，且未配置大模型 API Key。</p>
            </div>"""

def route_answer_with_context(history: List[Dict[str, str]], new_question: str, model_choice: str) -> tuple:
    """
    支持上下文历史的问答函数
    """
    # 获取当前对话历史
    conversation_history = history.copy()
    
    # 如果新问题为空，返回当前历史
    if not new_question or not new_question.strip():
        return conversation_history, ""
    
    question = new_question.strip()
    
    # 构建包含历史对话的上下文
    context = ""
    if len(conversation_history) > 0:
        context += "以下是之前的对话历史，本次回答请参考这些信息：\n"
        for i, item in enumerate(conversation_history[-3:], 1):  # 只取最近3轮对话
            context += f"Q{i}: {item['question']}\nA{i}: {item['answer']}\n\n"
        context += f"当前问题：{question}\n"
    else:
        context = question
    
    response = ""
    
    if model_choice == "本地农业分类模型":
        try:
            labels = predict(question)
            if labels:
                response = f"""<div style="background:#e8f5e8; border-left:4px solid #4caf50; padding:16px; border-radius:8px; margin:12px 0;">
                    <h3 style="color:#2e7d32; margin-top:0;">### 【分类结果】这个问题属于：{', '.join(labels)}</h3>
                </div>"""
            else:
                fallback = "\n\n💡 **提示**：未匹配到明确类别，建议切换至大模型获取详细解答。"
                response = f"""<div style="background:#fff3e0; border-left:4px solid #ff9800; padding:16px; border-radius:8px; margin:12px 0;">
                    <h3 style="color:#ef6c00; margin-top:0;">### 【分类结果】未匹配到任何预设类别。</h3>
                    <p>💡 <strong>提示</strong>：未匹配到明确类别，建议切换至大模型获取详细解答。</p>
                </div>"""
        except Exception as e:
            response = f"""<div style="background:#ffebee; border-left:4px solid #f44336; padding:16px; border-radius:8px; margin:12px 0;">
                <h3 style="color:#c62828; margin-top:0;">❌ 本地模型推理出错</h3>
                <p>{str(e)}</p>
            </div>"""

    elif model_choice == "智能路由模式":
        # 智能路由：先分类，再调用多个模型，最后整合回答
        labels = predict(question)
        if not labels:
            moonshot_resp = call_moonshot(context)
            response = f"""<div style="background:#e3f2fd; border-left:4px solid #2196f3; padding:16px; border-radius:8px; margin:12px 0;">
                <h3 style="color:#1565c0; margin-top:0;">💡 【智能路由】未匹配到明确类别，已使用 Moonshot 回答</h3>
                <div>{moonshot_resp}</div>
            </div>"""
        else:
            # 获取整合后的回答
            response = get_combined_answer(context, labels)

    elif model_choice == "Qwen 大模型":
        qwen_response = call_qwen(context)
        response = f"""<div style="background:#e0f2f1; border-left:4px solid #00bcd4; padding:16px; border-radius:8px; margin:12px 0;">
            <h3 style="color:#006064; margin-top:0;">### 【Qwen 回答】</h3>
            <div>{qwen_response}</div>
        </div>"""

    elif model_choice == "GLM 大模型":
        glm_response = call_glm(context)
        response = f"""<div style="background:#f3e5f5; border-left:4px solid #9c27b0; padding:16px; border-radius:8px; margin:12px 0;">
            <h3 style="color:#4a148c; margin-top:0;">### 【GLM 回答】</h3>
            <div>{glm_response}</div>
        </div>"""
    
    elif model_choice == "DeepSeek 大模型":
        deepseek_response = call_deepseek(context)
        response = f"""<div style="background:#f1f8e9; border-left:4px solid #8bc34a; padding:16px; border-radius:8px; margin:12px 0;">
            <h3 style="color:#33691e; margin-top:0;">### 【DeepSeek 回答】</h3>
            <div>{deepseek_response}</div>
        </div>"""
    
    elif model_choice == "Moonshot 大模型":
        moonshot_response = call_moonshot(context)
        response = f"""<div style="background:#e8eaf6; border-left:4px solid #3f51b5; padding:16px; border-radius:8px; margin:12px 0;">
            <h3 style="color:#283593; margin-top:0;">### 【Moonshot 回答】</h3>
            <div>{moonshot_response}</div>
        </div>"""
    
    else:
        response = """<div style="background:#ffebee; border-left:4px solid #f44336; padding:16px; border-radius:8px; margin:12px 0;">
            <h3 style="color:#c62828; margin-top:0;">❌ 未知模型选项</h3>
        </div>"""
    
    # 更新对话历史
    conversation_history.append({
        "question": question,
        "answer": response
    })
    
    # 清空输入框
    return conversation_history, ""

def format_chat_history(history: List[Dict[str, str]]) -> str:
    """
    格式化聊天历史为显示字符串
    """
    if not history:
        return "<div style='text-align: center; color: #888; padding: 20px;'>暂无对话历史</div>"
    
    formatted = "<div class='chat-container'>"
    for i, item in enumerate(history, 1):
        # 用户消息
        formatted += f"""
        <div class='message user-message'>
            <div class='message-content'><strong>用户:</strong> {item['question']}</div>
        </div>
        """
        # 助手消息
        formatted += f"""
        <div class='message assistant-message'>
            <div class='message-content'>{item['answer']}</div>
        </div>
        """
    formatted += "</div>"
    return formatted

def clear_history() -> tuple:
    """
    清空对话历史
    """
    return [], "", "对话历史已清空"

# 更新可用模型选项
ENHANCED_MODELS = ["本地农业分类模型", "智能路由模式"] + [m for m in AVAILABLE_MODELS if m != "本地农业分类模型"]

# 自定义CSS样式
custom_css = """
/* 整体样式 */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f5f7fa;
    margin: 0;
    padding: 0;
}

/* 主容器 */
.gradio-container {
    max-width: 100% !important;
    margin: 0 auto !important;
    padding: 20px !important;
    box-sizing: border-box;
}

/* 标题样式 */
h1 {
    text-align: center !important;
    color: #2c3e50 !important;
    font-size: 2em !important;
    margin-bottom: 5px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
}

h1::before {
    content: "🌱";
    font-size: 1.5em;
}

/* 描述文本 */
.description {
    text-align: center !important;
    color: #666 !important;
    font-size: 0.9em !important;
    margin-bottom: 20px !important;
}

/* 聊天容器 - 关键修改：可调整大小 */
.chat-container {
    min-height: 300px;
    max-height: 600px;      /* 可根据需要调大 */
    height: auto;           /* 关键！允许内容撑高 */
    overflow-y: auto;
    padding: 15px;
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    margin-bottom: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    resize: vertical;       /* 允许用户拖拽右下角调整高度 */
    /* 为 resize 提供视觉提示 */
    position: relative;
}
.chat-container::after {
    content: "";
    position: absolute;
    bottom: 4px;
    right: 4px;
    width: 12px;
    height: 12px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath fill='%23aaa' d='M8 6h3v3h-3zM8 1h3v3h-3zM1 1h3v3H1zM1 6h3v3H1zM1 11h3v3H1zM8 11h3v3h-3zM11 8h3v3h-3zM1 8h3v3H1z'/%3E%3C/svg%3E");
    background-size: 100%;
    cursor: nwse-resize;
    z-index: 1;
}

/* 消息样式 */
.message {
    margin-bottom: 15px;
    padding: 12px;
    border-radius: 8px;
    position: relative;
    word-wrap: break-word;
}

.user-message {
    background: #e3f2fd;
    border-left: 4px solid #2196f3;
    margin-left: 10px;
}

.assistant-message {
    background: #e8f5e8;
    border-left: 4px solid #4caf50;
    margin-right: 10px;
}

.message-content {
    line-height: 1.6;
}

.message-content h3, .message-content h4, .message-content h5 {
    margin: 10px 0 8px 0;
    color: #2c3e50;
}

.message-content ul, .message-content ol {
    padding-left: 20px;
    margin: 8px 0;
}

.message-content li {
    margin: 5px 0;
}

.message-content strong {
    font-weight: 600;
    color: #2c3e50;
}

.message-content code {
    background: #f8f9fa;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
}

.message-content pre {
    background: #f8f9fa;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 10px 0;
}

/* 输入区域 */
.input-area {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
    align-items: flex-end;
}

/* 增大输入框 */
.input-box-large {
    flex: 1;
    min-height: 80px;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 15px;
    font-size: 14px;
    resize: vertical;
}

/* 缩小提交按钮 */
.submit-btn-small {
    background: linear-gradient(45deg, #ff9800, #ff5722);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 12px 20px;
    cursor: pointer;
    font-weight: bold;
    transition: all 0.3s ease;
    min-width: 120px;
    max-width: 200px;
    white-space: nowrap;
}

.submit-btn-small:hover {
    background: linear-gradient(45deg, #ff8a00, #ff4500);
    transform: translateY(-1px);
}

/* 输入提示 */
.input-hint {
    text-align: left;
    color: #555;
    font-size: 0.9em;
    margin-bottom: 5px;
    padding-left: 10px;
}

/* 操作按钮区域 */
.action-buttons {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
    background: #f8f9fa;
    padding: 10px;
    border-radius: 6px;
    border: 1px solid #ddd;
}

.action-button {
    flex: 1;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 8px 12px;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s ease;
    font-size: 0.9em;
}

.action-button:hover {
    background: #e9ecef;
    border-color: #ccc;
}

/* 示例区域 */
.examples-section {
    margin-bottom: 15px;
    background: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 15px;
}

.examples-title {
    font-weight: bold;
    margin-bottom: 10px;
    color: #2c3e50;
    display: flex;
    align-items: center;
    gap: 5px;
}

.example-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
}

.example-table th, .example-table td {
    padding: 8px;
    text-align: left;
    border: 1px solid #eee;
}

.example-table th {
    background: #f8f9fa;
    font-weight: bold;
}

.example-table tr:nth-child(even) {
    background: #fafafa;
}

/* 模型选择区域 */
.model-selector {
    background: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 15px;
    margin-bottom: 15px;
}

.model-selector-title {
    font-weight: bold;
    margin-bottom: 10px;
    color: #2c3e50;
    display: flex;
    align-items: center;
    gap: 5px;
}

.model-options {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.model-option {
    background: #f8f9fa;
    border: 1px solid #ddd;
    border-radius: 20px;
    padding: 8px 15px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0.9em;
}

.model-option:hover {
    background: #e9ecef;
    border-color: #ccc;
}

.model-option.selected {
    background: #4CAF50;
    color: white;
    border-color: #4CAF50;
}

/* 滚动条样式 */
.chat-container::-webkit-scrollbar {
    width: 6px;
}

.chat-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

.chat-container::-webkit-scrollbar-thumb {
    background: #ddd;
    border-radius: 4px;
}

.chat-container::-webkit-scrollbar-thumb:hover {
    background: #bbb;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .input-area {
        flex-direction: column;
    }
    
    .model-options {
        flex-direction: column;
    }
    
    .chat-container {
        height: 250px;
    }
    
    .submit-btn-small {
        min-width: auto;
        max-width: 100%;
    }
}

/* 专为整合报告设计的样式 */
.answer-section {
    padding: 10px;
    background: #f9fbfd;
    border-radius: 6px;
    border-left: 3px solid #2196f3;
    margin-bottom: 12px;
}

.answer-section strong {
    color: #1a237e;
    font-weight: 600;
}

h3 {
    color: #2e7d32 !important;
    margin: 18px 0 10px 0 !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 6px;
}

ul, ol {
    padding-left: 24px;
    margin: 10px 0;
}

li {
    margin: 6px 0;
}

li::before {
    content: "✅ ";
    color: #2e7d32;
    font-weight: bold;
    display: inline-block;
    width: 20px;
}

/* 高亮关键术语 */
.highlight {
    background: #e8f5e8;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
}
.chat-placeholder {
    width: 100%;
}
/* 重要：为聊天容器添加调整手柄 */

"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        # 标题区
        gr.Markdown(
            "# 🌱 农业智能体 - 多轮专业对话系统",
            elem_classes="title"
        )
        
        # 描述区
        gr.Markdown(
            "支持上下文记忆、智能标签路由、多模型协作回答。推荐使用「智能路由模式」以获得最匹配的农业专业解答。",
            elem_classes="description"
        )
        
        # 聊天区域
        chat_display = gr.HTML(
            label="Chatbot",
            value='<div class="chat-placeholder" style="min-height: 280px; display:flex; align-items:center; justify-content:center; color:#999; font-size:0.9em;">开始您的对话吧！</div>',
            elem_classes="chat-container"
)
        
        # 操作按钮区域
        with gr.Row(elem_classes="action-buttons"):
            retry_btn = gr.Button("🔄 Retry", variant="secondary", elem_classes="action-button")
            undo_btn = gr.Button("↩️ Undo", variant="secondary", elem_classes="action-button")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary", elem_classes="action-button")
        
        # 输入提示
        gr.Markdown(
            "💡 请输入您的农业问题（例如：小麦白粉病防治方法）",
            elem_classes="input-hint"
        )
        
        # 输入区域
        with gr.Row(elem_classes="input-area"):
            # 关键修改：增加高度和合理空间分配
            input_box = gr.Textbox(
                label="",
                placeholder="在此输入您的问题...",
                lines=3,
                elem_classes="input-box-large"
            )
            # 关键修改：限制按钮宽度
            submit_btn = gr.Button("Submit", elem_classes="submit-btn-small")
        
        # 示例区域
        gr.Markdown("### 💡 Examples", elem_classes="examples-title")
        with gr.Column(elem_classes="examples-section"):
            gr.HTML("""
            <table class="example-table">
                <thead>
                    <tr>
                        <th>Message</th>
                        <th>回答模式</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>如何防治苹果树腐烂病？</td>
                        <td>智能路由模式</td>
                    </tr>
                    <tr>
                        <td>什么是光合作用？</td>
                        <td>智能路由模式</td>
                    </tr>
                    <tr>
                        <td>今年晚稻追肥应该注意什么？</td>
                        <td>智能路由模式</td>
                    </tr>
                    <tr>
                        <td>小麦白粉病早期症状有哪些？</td>
                        <td>智能路由模式</td>
                    </tr>
                </tbody>
            </table>
            """)
        
        # 模型选择区域
        gr.Markdown("### ⚙️ Additional Inputs", elem_classes="model-selector-title")
        with gr.Column(elem_classes="model-selector"):
            gr.Markdown("#### 回答模式", elem_classes="model-selector-title")
            model_choice = gr.Radio(
                choices=ENHANCED_MODELS,
                label="",
                value="智能路由模式",
                interactive=True,
                elem_classes="model-options"
            )
    
    # 状态变量
    chat_history = gr.State([])
    
    # 绑定事件
    submit_btn.click(
        fn=route_answer_with_context,
        inputs=[chat_history, input_box, model_choice],
        outputs=[chat_history, input_box]
    ).then(
        fn=format_chat_history,
        inputs=[chat_history],
        outputs=[chat_display]
    )
    
    clear_btn.click(
        fn=clear_history,
        inputs=None,
        outputs=[chat_history, input_box, chat_display]
    )

if __name__ == "__main__":
    print("🚀 启动农业智能体 Web 界面...")
    print(f"可用模型: {AVAILABLE_MODELS}")
    print(f"标签到模型映射: {LABEL_TO_MODEL}")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)