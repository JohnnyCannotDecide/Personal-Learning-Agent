"""
import openai
client = openai.OpenAI(
  api_key="sk-hLC8c2HA5B7WVjUo743fDd06Eb6a4768Ac41980cE017C0Bc", 
  base_url="https://aihubmix.com/v1"
)
response = client.chat.completions.create(
  model="gpt-4.1-free",
  messages=[
      {"role": "user", "content": "你好，请介绍一下你自己"}
  ]
)
print(response.choices[0].message.content)



import openai

# 初始化客户端
client = openai.OpenAI(
    api_key="sk-hLC8c2HA5B7WVjUo743fDd06Eb6a4768Ac41980cE017C0Bc", 
    base_url="https://aihubmix.com/v1"
)

# 1. 创建一个列表来存储对话历史
# 你可以预设一个 system 角色，告诉模型它是谁
messages = [
    {"role": "system", "content": "你是一个乐于助人的 AI 助手。"}
]
print("--- 已进入对话模式（输入 '退出' 结束程序） ---")
while True:
    # 2. 获取用户输入
    user_input = input("\n用户: ").strip()
    # 3. 检查退出条件
    if user_input.lower() in ["退出", "exit", "quit"]:
        print("AI: 再见！祝你有愉快的一天。")
        break
    if not user_input:
        continue
    # 4. 将用户的输入添加到对话历史中
    messages.append({"role": "user", "content": user_input})
    try:
        # 5. 发送请求（注意这里发送的是整个 messages 列表）
        response = client.chat.completions.create(
            model="gpt-4.1-free",
            messages=messages,
            stream=False # 如果需要打字机效果，可以改为 True 并在下方处理流式输出
        )

        # 6. 获取模型回复
        assistant_message = response.choices[0].message
        print(f"AI: {assistant_message.content}")

        # 7. 关键一步：将模型的回复也存入历史，以便下一轮对话参考
        messages.append(assistant_message)
    except Exception as e:
        print(f"发生错误: {e}")
        
print(messages)

import openai

# 1. 初始化客户端
client = openai.OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

def check_sentiment(text):
    
    #判断输入内容是否包含负面情绪
    #返回 True (负面) 或 False (非负面)
    
    try:
        response = client.chat.completions.create(
            model="openrouter/hunter-alpha",
            messages=[
                {"role": "system", "content": "你是一个情绪识别专家。请判断用户的话语是否包含负面情绪（如悲伤、愤怒、沮丧、孤独等）。如果是，请只回答'YES'，否则只回答'NO'。不要说任何多余的话。"},
                {"role": "user", "content": text}
            ],
            temperature=0 # 设为 0 增加判断的稳定性
        )
        result = response.choices[0].message.content.strip().upper()
        return "YES" in result
    except Exception:
        return False # 如果判断出错，默认不触发提示，确保程序继续运行

# 初始化对话历史
messages = [
    {"role": "system", "content": "你是一个乐于助人的 AI 助手。"}
]

print("--- 已进入情绪感知对话模式（输入 '退出' 结束） ---")

while True:
    user_input = input("\n用户: ").strip()
    
    if user_input.lower() in ["退出", "exit", "quit"]:
        print("AI: 再见！")
        break
    
    if not user_input:
        continue

    # --- 新增功能：情绪检测 ---
    if check_sentiment(user_input):
        print("\n[系统感知] AI: 我注意到你可能心情不太好，我在这里。")

    # --- 正常对话流程 ---
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="openrouter/hunter-alpha",
            messages=messages
        )

        assistant_message = response.choices[0].message
        print(f"AI: {assistant_message.content}")

        # 将回复存入记忆
        messages.append(assistant_message)

    except Exception as e:
        print(f"对话发生错误: {e}")


from openai import OpenAI
import json
import time

# 1. 初始化客户端
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

# 工具内部接入 API 请求 ---
def check_emotion_tool(text):

    #调用专门的情绪识别 API

    print(f"\n[系统日志] 正在启动独立 AI 专家分析情绪...")
    try:
        # 这里的请求是独立的，不会干扰主对话的记忆
        response = client.chat.completions.create(
            model="openrouter/hunter-alpha", # 使用更轻快且支持工具的模型
            messages=[
                {"role": "system", "content": "你是一个情感分析专家。请分析用户的文本，判断其情感极性。如果包含明显的负面情绪（悲伤、愤怒、极度焦虑），返回 JSON: {'is_negative': true, 'reason': '原因简述'}。否则返回 {'is_negative': false, 'reason': '正常'}"},
                {"role": "user", "content": text}
            ],
            response_format={ "type": "json_object" } # 强制返回 JSON 格式
        )
        # 解析 AI 专家的判断结果
        sentiment_result = json.loads(response.choices[0].message.content)
        return sentiment_result
    except Exception as e:
        print(f"[错误] 情绪分析 API 异常: {e}")
        return {"is_negative": False, "reason": "检测失败"}

# 2. 工具定义（保持不变）
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_emotion",
            "description": "当用户语气中流露出情感波动、不悦或悲伤时，调用此工具核实情绪状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_text": { "type": "string", "description": "用户的原始输入内容" }
                },
                "required": ["user_text"]
            }
        }
    }
]

# 3. 主对话逻辑
messages = [
    {"role": "system", "content": "你是一个高情商的 AI 助手。如果用户心情不好，请先调用 check_emotion 工具确认。若确认为负面情绪，请先给出一句温暖的安慰语，再正式回答。"}
]

print("--- Agent 模式启动（已接入 AI 情绪专家） ---")

while True:
    user_input = input("\n用户: ").strip()
    if user_input.lower() in ["退出", "exit", "quit"]: break
    if not user_input: continue

    messages.append({"role": "user", "content": user_input})

    # 第一轮请求：判断是否需要调用工具
    response = client.chat.completions.create(
        model="openrouter/hunter-alpha",
        messages=messages,
        tools=tools
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        # 模型决定调用工具
        messages.append(response_message)
        for tool_call in tool_calls:
            if tool_call.function.name == "check_emotion":
                args = json.loads(tool_call.function.arguments)
                # --- 这里执行了真正的 API 请求 ---
                result = check_emotion_tool(args.get("user_text"))
                
                # 手动打印前置提示（模拟你的需求）
                if result.get("is_negative"):
                    print("\n[AI 感知到你的情绪：我注意到你可能心情不太好，我在这里。]")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "check_emotion",
                    "content": json.dumps(result)
                })

        # 第二轮请求：整合工具结果生成最终回复
        second_response = client.chat.completions.create(
            model="openrouter/hunter-alpha",
            messages=messages
        )
        final_answer = second_response.choices[0].message.content
        print(f"AI: {final_answer}")
        messages.append({"role": "assistant", "content": final_answer})
    else:
        # 无需工具，正常回复
        print(f"AI: {response_message.content}")
        messages.append({"role": "assistant", "content": response_message.content})
"""

import os
import json
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# 1. 配置 OpenRouter (保持不变)
OPENROUTER_API_KEY = "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"
llm = ChatOpenAI(
    model="openrouter/hunter-alpha",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)
# 2. LangChain 对话记忆
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
# 3. 情绪检测 Tool
def check_emotion_tool(text: str):
    print("\n[系统日志] 正在请求 OpenRouter 专家模型分析情绪...")
    emotion_llm = ChatOpenAI(
        model="openrouter/hunter-alpha",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )
    prompt = f"""你是一个情感分析专家。分析下面用户文本的情绪：{text}如果包含负面情绪返回：{{"is_negative": true}}否则返回：{{"is_negative": false}}只返回JSON
"""
    try:
        response = emotion_llm.invoke(prompt)
        result = json.loads(response.content)
        return result
    except Exception as e:
        print(f"情绪检测失败: {e}")
        return {"is_negative": False}
# 4. 主对话循环
print("AI助手已启动（输入 exit 退出）")
while True:
    user_input = input("\n用户: ").strip()
    if user_input.lower() in ["exit", "退出"]:
        break
    try:
        # 1️⃣ 先进行情绪检测
        sentiment = check_emotion_tool(user_input)
        if sentiment.get("is_negative"):
            print("\n[AI感知] 我注意到你可能心情不太好，我在这里。")
        # 2️⃣ 使用 ConversationChain 进行对话
        reply = conversation.predict(input=user_input)
        print(f"\nAI: {reply}")
    except Exception as e:
        print(f"OpenRouter 连接错误: {e}")