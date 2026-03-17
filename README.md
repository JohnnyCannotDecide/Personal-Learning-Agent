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
