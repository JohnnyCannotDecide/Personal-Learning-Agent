from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, stream_with_context
from learning_tracker import PersonalBrainAgent
import os
import json
from openai import OpenAI

app = Flask(__name__)

# 初始化 Agent
agent = PersonalBrainAgent()
openai_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def build_prompt_from_query(query, top_k=3):
    results = agent.search(query, top_k)
    if not results:
        context = "无相关知识"
    else:
        context = "\n".join([
            f"[{r['date']}] {r['title']} - {r['content']}"
            for r in results
        ])
    return f"""
你是一个个人知识助手，请基于以下记录回答问题：

【历史记录】
{context}

【问题】
{query}

要求：
1. 优先基于记录回答
2. 如果信息不足，可以补充
"""

def stream_llm(query):
    prompt = build_prompt_from_query(query)
    stream = openai_client.chat.completions.create(
        model="google/gemma-3-12b-it:free",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {delta}\n\n"
    yield "data: [DONE]\n\n"

# 注入用户背景（检查是否已存在，避免重复添加）
def inject_user_background():
    background_title = "用户背景"
    # 检查 metadata 中是否已经有这条记录
    exists = any(item['title'] == background_title for item in agent.metadata)
    if not exists:
        agent.add({
            "date": "2026/03/17",
            "title": background_title,
            "content": "用户是一个刚毕业的大学生，正在学习Agent应用开发，熟悉Python基础，已完成LLM调用、多轮对话、Function Calling、LangGraph、RAG的学习，目标是找到Agent应用开发的工作，倾向于在别人的模型上构建应用而不是训练模型"
        })

inject_user_background()

ADVICE_FILE = os.path.join(os.path.dirname(__file__), "advice.json")

def load_advice_map():
    if not os.path.exists(ADVICE_FILE):
        return {}
    try:
        with open(ADVICE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_advice_map(data):
    with open(ADVICE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "Query is empty"}), 400
    
    # 调用你定义的 ask 方法
    response = agent.ask(query)
    return jsonify({"response": response})

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    data = request.json or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({"error": "Query is empty"}), 400
    return Response(stream_with_context(stream_llm(query)), mimetype='text/event-stream')

@app.route('/search_eval', methods=['GET'])
def search_eval():
    # 目的：提供一个可直接调用的检索评估入口，便于持续验证 Precision@3。
    result = agent.evaluate_search_precision_at_3()
    return jsonify(result)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json or {}
        node_name = (data.get('node_name') or '').strip()
        history = data.get('history') or []
        user_answer = (data.get('user_answer') or '').strip()
        if not node_name:
            return jsonify({"error": "node_name is required"}), 400

        context_records = agent.search(f"{node_name} 学习记录 评估", top_k=5)
        if context_records:
            context = "\n".join([f"[{r['date']}] {r['title']} - {r['content']}" for r in context_records])
        else:
            context = "无相关学习记录"

        history_lines = []
        if isinstance(history, list):
            for item in history:
                role = (item.get('role') or '').strip()
                content = (item.get('content') or '').strip()
                if role and content:
                    history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "（无）"
        answer_text = user_answer if user_answer else "（首轮，请直接出一道该技术的核心概念题）"

        prompt = f"""
    你是一个严格但友善的技术面试官，正在评估用户对"{node_name}"的掌握程度。 
    规则： 
    1. 第一次调用时，出一道该技术的核心概念题，不要太简单 
    2. 根据用户回答决定：回答到位就追问更深，回答不全就指出缺失并继续追问 
    3. 经过2-4轮对话后，给出评估结论 
    4. 结论必须包含这个标记之一：[建议继续学习] 或 [基本掌握] 
    5. 说话简洁，像真实面试官，不要啰嗦 
    你的提问必须严格限定在{node_name}这个知识点本身，不要延伸到其他模块或生产级工程化细节，评估目标是概念理解和基本实践能力。

    学习记录参考：
    {context}

    已有对话历史：
    {history_text}

    用户本轮回答：
    {answer_text}

    请直接输出你这一轮要说的话，不要输出多余说明。
    """
        response = agent.llm.invoke(prompt)
        model_reply = response.content if hasattr(response, 'content') else str(response)
        passed = '[基本掌握]' in model_reply
        return jsonify({"response": model_reply, "passed": passed})
    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整堆栈到终端
        return jsonify({"error": str(e)}), 500
    
@app.route('/save_advice', methods=['POST'])
def save_advice():
    data = request.json or {}
    node_name = (data.get('node_name') or '').strip()
    advice = (data.get('advice') or '').strip()
    if not node_name or not advice:
        return jsonify({"error": "node_name and advice are required"}), 400
    advice_map = load_advice_map()
    advice_map[node_name] = advice
    save_advice_map(advice_map)
    return jsonify({"status": "success"})

@app.route('/get_advice', methods=['GET'])
def get_advice():
    node_name = (request.args.get('node') or '').strip()
    if not node_name:
        return jsonify({"error": "node is required"}), 400
    advice_map = load_advice_map()
    advice = advice_map.get(node_name, "")
    return jsonify({"found": bool(advice), "advice": advice})

@app.route('/add', methods=['POST'])
def add_record():
    data = request.json
    new_record = {
        "date": data.get('date'),
        "title": data.get('title'),
        "content": data.get('content')
    }
    agent.add(new_record)
    return jsonify({"status": "success"})

@app.route('/upload', methods=['POST'])
def upload_document():
    data = request.json or {}
    title = (data.get('title') or '').strip()
    content = (data.get('content') or '').strip()
    
    if not title or not content:
        return jsonify({"error": "title and content required"}), 400
    
    chunk_count = agent.add_document(title, content)
    return jsonify({
        "status": "success",
        "chunks": chunk_count,
        "message": f"已切分为 {chunk_count} 个片段并存入知识库"
    })
    
@app.route('/timeline', methods=['GET'])
def get_timeline():
    # 给每一条记录打上原始索引标签，解决排序后的索引错位问题
    data_with_id = []
    for i, item in enumerate(agent.metadata):
        copy_item = item.copy()
        copy_item['original_idx'] = i
        data_with_id.append(copy_item)
    # 按日期倒序排列
    return jsonify(sorted(data_with_id, key=lambda x: x['date'], reverse=True))

@app.route('/record', methods=['DELETE'])
def delete_record():
    idx = request.json.get('index')
    # 注意：前端传来的 index 可能是基于时间轴排序后的，
    # 建议前端直接传原始 metadata 的索引，或者通过 title/date 匹配。
    # 这里我们假设传的是原始索引
    if agent.delete(idx):
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/map')
def skill_map():
    # 返回技能树主页面
    return render_template('map.html')

@app.route('/tracker')
def tracker():
    return render_template('index.html')

@app.route('/')
def index():
    return redirect(url_for('skill_map'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
