# Personal Learning Agent

An interview-ready project that demonstrates my end-to-end Agent application capability: product thinking, full-stack implementation, memory retrieval, and interactive evaluation workflow.

## Why This Project Matters
Personal Learning Agent is a RAG-based learning system with a visual skill tree and Socratic evaluation loop.  
It tracks learning milestones, retrieves personal context from vector memory, and generates targeted guidance per node.  
Instead of a static demo, it forms a complete user loop: **plan → learn → assess → update progress**.

## Recruiter Highlights
- **End-to-end ownership**: UI interaction, backend APIs, data persistence, and retrieval pipeline
- **Agent-oriented design**: node-level evaluation with multi-turn questioning and pass/fail signal
- **Practical RAG**: recommendations grounded in personal learning history, not generic output
- **Productized workflow**: map entry, tracker workspace, persistent suggestions, and state transitions
- **Execution speed**: built from zero during an intensive 4-day learning sprint

## Core Features
- **Skill Tree Map**
  - Visual dependency graph of learning topics
  - Node states: `completed`, `active`, `locked`
  - Right-side detail panel with status, criteria, and actions
- **Node Evaluation Mode**
  - Full-screen interview-style assessment
  - Multi-turn follow-up questions
  - If model returns `[基本掌握]`, user can mark node as completed
- **Learning Tracker**
  - Timeline of milestones with add/delete persistence
  - Chat workspace connected to personal knowledge base
- **RAG Guidance**
  - Generates suggestions from personal records via FAISS retrieval
  - Node-level advice cache with reload and regeneration flow

## Architecture Snapshot
- **Frontend**: `map.html` + `index.html` (skill map, detail drawer, evaluation modal, tracker UI)
- **Backend**: Flask routes for chat, evaluate, timeline CRUD, and advice persistence
- **Memory Layer**: FAISS vector index + metadata JSON for retrieval context
- **LLM Layer**: OpenAI-compatible chat model and embedding model through LangChain

## Tech Stack
- Python
- Flask
- LangChain
- FAISS
- OpenAI-compatible APIs

## Quick Start
1. **Clone**
   ```bash
   git clone <your-repo-url>
   cd agent-learning
   ```

2. **Set up environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install flask langchain-openai faiss-cpu numpy
   ```

4. **Configure API keys**
   - Open `learning_tracker.py`
   - Configure:
     - chat model API key + base URL
     - embedding model API key + base URL

5. **Run**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000`

## Project Structure
- `app.py`: Flask app entry, routing, evaluation API, advice persistence API
- `learning_tracker.py`: memory agent, embedding generation, FAISS search, answer generation
- `templates/map.html`: skill map UI, node panel, advice and evaluation interactions
- `templates/index.html`: tracker workspace and timeline management
- `learning_db/`: vector index and metadata persistence
- `advice.json`: cached node-level advice
- `test.py`: local experimentation script

## Development Background
I built this project on the **fourth day** of my Agent development journey.  
Starting from zero, I used four days to move from basic LLM API calls to a complete Agent application with retrieval memory, multi-page workflow, and interview-style evaluation logic.  
This system is built with the same techniques I learned during that process, and it also records that process as its own data source.

## Optional Demo Add-ons
- Add screenshots/GIFs for:
  - skill map with node states
  - evaluation modal flow
  - tracker timeline and RAG response
- Add a short 2-minute walkthrough video link for interview sharing

---

## 中文

### 项目简介
Personal Learning Agent 是一个基于 RAG 与 LangGraph 思路构建的个人学习追踪系统。它通过技能树可视化学习路径，让每个知识点的依赖关系和进度状态清晰可见。系统引入苏格拉底式追问评估，检验是否真正掌握，而不是只做被动打卡。它还会结合个人学习记录进行检索，生成更有针对性的学习建议。

### 功能特性
- **技能树地图**
  - 用依赖连线展示学习路径
  - 支持三种节点状态：`completed`、`active`、`locked`
- **节点评估**
  - 面向单一知识点的追问式评估
  - 通过后可将节点标记为完成
- **学习追踪**
  - 时间轴记录学习里程碑
  - 学习记录持久化存储
- **RAG 检索建议**
  - 建议基于个人学习记录生成，而非泛化回答
- **建议持久化**
  - 每个节点建议自动缓存并复用
  - 需要时可手动“重新生成”

### 技术栈
- Python
- Flask
- LangChain
- FAISS
- OpenAI 兼容 API

### 快速开始
1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd agent-learning
   ```

2. **创建并激活虚拟环境**
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```

3. **安装依赖**
   ```bash
   pip install flask langchain-openai faiss-cpu numpy
   ```

4. **配置 API Key**
   - 打开 `learning_tracker.py`
   - 配置 OpenAI 兼容接口的：
     - 大模型 API 信息（如 OpenRouter 风格 endpoint）
     - Embedding API 信息

5. **启动项目**
   ```bash
   python app.py
   ```
   浏览器访问 `http://127.0.0.1:5000`。

### 项目结构
- `app.py`  
  Flask 入口与路由层（`/map`、`/tracker`、`/chat`、`/evaluate`、建议持久化接口）。
- `learning_tracker.py`  
  个人记忆 Agent 核心逻辑：embedding、FAISS 检索与回答生成。
- `templates/map.html`  
  技能树页面、节点详情面板、评估窗口与建议交互。
- `templates/index.html`  
  学习追踪工作台（聊天 + 时间轴管理）。
- `learning_db/`  
  向量索引与元数据持久化目录（`vector_index.faiss`、`metadata.json`）。
- `advice.json`  
  节点级学习建议缓存文件。
- `test.py`  
  本地测试与快速实验脚本。

### 开发背景
这个项目是我在学习 Agent 开发的第 4 天搭建的。从零开始，我在 4 天内完成了从 LLM API 调用到完整 Agent 应用的学习和落地。这个系统本身就是我用所学技术构建的，也真实记录了我的学习过程。

### 页面显示

路线图

<img width="1641" height="576" alt="image" src="https://github.com/user-attachments/assets/f1f237df-1d1e-4bc6-adff-4ed7f1b6efc4" />

学习建议（建议结果可保存）

<img width="403" height="644" alt="image" src="https://github.com/user-attachments/assets/692abbdb-5344-4387-a407-2d4e0575923a" />

过程记录

<img width="1907" height="824" alt="image" src="https://github.com/user-attachments/assets/38250b51-d618-409d-a8bb-4a79959d5558" />


