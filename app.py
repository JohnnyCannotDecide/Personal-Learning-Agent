from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, stream_with_context
from learning_tracker import PersonalBrainAgent
import os
import json
from openai import OpenAI
import fitz
import warnings
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")
app = Flask(__name__)

# 初始化 Agent
agent = PersonalBrainAgent()
openrouter_api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
openai_client = OpenAI(
    api_key=openrouter_api_key,
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


def resolve_node(node_name, node_id):
    tree = agent.get_skill_tree_view()
    if node_id:
        for node in tree.get("nodes", []):
            if node.get("id") == node_id:
                return node
    if node_name:
        lower_name = node_name.lower()
        for node in tree.get("nodes", []):
            if (node.get("label") or "").lower() == lower_name:
                return node
    return None

def stream_llm(query):
    if not openrouter_api_key:
        yield "data: 系统错误：未配置 OPENROUTER_API_KEY，请先在 .env 配置有效密钥。\n\n"
        yield "data: [DONE]\n\n"
        return
    prompt = build_prompt_from_query(query)
    try:
        stream = openai_client.chat.completions.create(
            model="qwen/qwen3.6-plus-preview:free",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {delta}\n\n"
    except Exception as e:
        error_text = str(e)
        if "401" in error_text or "AuthenticationError" in error_text:
            yield "data: 认证失败：OPENROUTER_API_KEY 无效或已过期，请更换有效密钥。\n\n"
        else:
            yield f"data: 流式请求失败：{error_text}\n\n"
    finally:
        yield "data: [DONE]\n\n"

def extract_pdf_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

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
        node_id = (data.get('node_id') or '').strip()
        history = data.get('history') or []
        user_answer = (data.get('user_answer') or '').strip()
        node = resolve_node(node_name, node_id)
        if not node:
            return jsonify({"error": "node_name or node_id is invalid"}), 400
        result = agent.evaluate_node(node_id=node.get("id"), history=history, user_answer=user_answer)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/save_advice', methods=['POST'])
def save_advice():
    data = request.json or {}
    node_name = (data.get('node_name') or '').strip()
    node_id = (data.get('node_id') or '').strip()
    advice = (data.get('advice') or '').strip()
    if not advice:
        return jsonify({"error": "advice is required"}), 400
    node = resolve_node(node_name, node_id)
    if not node:
        return jsonify({"error": "node_name or node_id is invalid"}), 400
    key = node.get("id")
    advice_map = load_advice_map()
    advice_map[key] = advice
    save_advice_map(advice_map)
    return jsonify({"status": "success"})

@app.route('/get_advice', methods=['GET'])
def get_advice():
    node_name = (request.args.get('node') or '').strip()
    node_id = (request.args.get('node_id') or '').strip()
    node = resolve_node(node_name, node_id)
    if not node:
        return jsonify({"error": "node or node_id is required"}), 400
    advice_map = load_advice_map()
    advice = advice_map.get(node.get("id"), "")
    return jsonify({"found": bool(advice), "advice": advice})

@app.route('/add', methods=['POST'])
def add_record():
    data = request.json or {}
    node_id = (data.get('node_id') or '').strip()
    new_record = {
        "date": data.get('date'),
        "title": data.get('title'),
        "content": data.get('content'),
        "skill_id": node_id
    }
    agent.add(new_record)
    return jsonify({"status": "success"})

@app.route('/upload', methods=['POST'])
def upload_document():
    source = "upload"
    if request.files and 'file' in request.files:
        file = request.files['file']
        filename = file.filename or ''
        title = (request.form.get('title') or '').strip() or os.path.splitext(filename)[0]
        is_pdf = (file.mimetype or '').lower() == 'application/pdf' or filename.lower().endswith('.pdf')
        raw_bytes = file.read()
        if is_pdf:
            content = extract_pdf_text(raw_bytes).strip()
            source = "pdf"
        else:
            content = raw_bytes.decode('utf-8', errors='ignore').strip()
            source = "upload"
    else:
        data = request.json or {}
        title = (data.get('title') or '').strip()
        content = (data.get('content') or '').strip()
        node_id = (data.get('node_id') or '').strip()
        source = "upload"
    if request.files and 'file' in request.files:
        node_id = (request.form.get('node_id') or '').strip()
    if not title or not content:
        return jsonify({"error": "title and content required"}), 400
    chunk_count = agent.add_document(title, content, source=source, skill_id=node_id or None)
    return jsonify({
        "status": "success",
        "chunks": chunk_count,
        "message": f"已切分为 {chunk_count} 个片段并存入知识库"
    })
    
@app.route('/timeline', methods=['GET'])
def get_timeline():
    data_with_id = []
    for i, item in enumerate(agent.metadata):
        copy_item = item.copy()
        copy_item['original_idx'] = i
        data_with_id.append(copy_item)
    return jsonify(sorted(data_with_id, key=lambda x: x['date'], reverse=True))

@app.route('/record', methods=['DELETE'])
def delete_record():
    idx = request.json.get('index')
    if agent.delete(idx):
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/map')
def skill_map():
    return render_template('map.html')


@app.route('/skills/tree', methods=['GET'])
def skill_tree_data():
    return jsonify(agent.get_skill_tree_view())


@app.route('/skills/consult', methods=['POST'])
def skill_consult():
    data = request.json or {}
    query = (data.get('query') or '').strip()
    node_id = (data.get('node_id') or '').strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    result = agent.consult(query=query, node_id=node_id or None)
    return jsonify(result)


@app.route('/skills/advice', methods=['POST'])
def skill_advice():
    data = request.json or {}
    node_id = (data.get('node_id') or '').strip()
    if not node_id:
        return jsonify({"error": "node_id is required"}), 400
    advice_map = load_advice_map()
    if not data.get('force_regenerate') and advice_map.get(node_id):
        return jsonify({"response": advice_map[node_id], "cached": True, "references": []})
    result = agent.advise(node_id=node_id)
    advice_map[node_id] = result.get("response", "")
    save_advice_map(advice_map)
    result["cached"] = False
    return jsonify(result)


@app.route('/skills/evaluate', methods=['POST'])
def skill_evaluate():
    data = request.json or {}
    node_id = (data.get('node_id') or '').strip()
    history = data.get('history') or []
    user_answer = (data.get('user_answer') or '').strip()
    if not node_id:
        return jsonify({"error": "node_id is required"}), 400
    return jsonify(agent.evaluate_node(node_id=node_id, history=history, user_answer=user_answer))


@app.route('/skills/complete', methods=['POST'])
def skill_complete():
    data = request.json or {}
    node_id = (data.get('node_id') or '').strip()
    if not node_id:
        return jsonify({"error": "node_id is required"}), 400
    if not agent.mark_skill_completed(node_id):
        return jsonify({"error": "node_id not found"}), 404
    return jsonify({"status": "success"})

@app.route('/tracker')
def tracker():
    return render_template('index.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/')
def index():
    return redirect(url_for('landing'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
