import os
import json
import faiss
import numpy as np
import math
import re
from collections import Counter
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


class PersonalBrainAgent:
    def __init__(self, db_dir="learning_db", skill_tree_file="skill_tree_agent.json"):
        self.db_dir = db_dir
        self.index_file = os.path.join(db_dir, "vector_index.faiss")
        self.meta_file = os.path.join(db_dir, "metadata.json")
        self.progress_file = os.path.join(db_dir, "skill_progress.json")
        self.skill_tree_file = os.path.join(os.path.dirname(__file__), skill_tree_file)

        os.makedirs(self.db_dir, exist_ok=True)
        load_dotenv()

        self.llm = ChatOpenAI(
            model="qwen/qwen3.6-plus-preview:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
        )

        self.embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-8B",
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        print("已加载 API 模型（LLM + Embedding）")

        self.reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
        self.skill_tree = self._load_skill_tree()

        self._load_or_create_db()
        self.skill_progress = self._load_skill_progress()

    def _default_skill_tree(self):
        return {
            "track_id": "agent_learning_v1",
            "name": "Agent 学习路线",
            "retrieval_eval_cases": [
                {"query": "我该怎么定位Agent方向的职业目标？", "expected_title": "市场锚定"},
                {"query": "我需要补强简历表达能力，应该回顾什么？", "expected_title": "简历分析"},
                {"query": "怎么理解多轮对话的上下文管理？", "expected_title": "接口实战"},
                {"query": "LangGraph 的核心编排思想是什么？", "expected_title": "架构初探"},
                {"query": "RAG 怎么降低模型胡说八道？", "expected_title": "数据增强"}
            ],
            "nodes": [
                {
                    "id": "llm_api",
                    "label": "LLM API 调用",
                    "x": 100,
                    "y": 300,
                    "link": "/tracker",
                    "criteria": "能够稳定调用主流大模型 API，并完成基础参数配置与错误处理。",
                    "keywords": ["llm", "api", "调用", "openrouter", "模型接口"],
                    "prerequisites": [],
                    "module": "咨询"
                },
                {
                    "id": "multi_turn",
                    "label": "多轮对话",
                    "x": 300,
                    "y": 300,
                    "criteria": "可以维护多轮上下文，让连续追问下的回答保持一致性。",
                    "keywords": ["多轮对话", "上下文", "history", "chat"],
                    "prerequisites": ["llm_api"],
                    "module": "咨询"
                },
                {
                    "id": "function_calling",
                    "label": "Function Calling",
                    "x": 500,
                    "y": 300,
                    "criteria": "能够按 schema 调用工具并处理参数校验与异常回退。",
                    "keywords": ["function calling", "工具调用", "schema", "参数校验"],
                    "prerequisites": ["multi_turn"],
                    "module": "咨询"
                },
                {
                    "id": "langgraph",
                    "label": "LangGraph 编排",
                    "x": 700,
                    "y": 200,
                    "criteria": "可以设计状态流并解释每个节点职责。",
                    "keywords": ["langgraph", "状态机", "编排", "节点"],
                    "prerequisites": ["function_calling"],
                    "module": "建议"
                },
                {
                    "id": "rag",
                    "label": "RAG 检索",
                    "x": 700,
                    "y": 400,
                    "criteria": "能完成检索、重排和生成的基础链路并验证效果。",
                    "keywords": ["rag", "检索", "重排", "向量", "bm25"],
                    "prerequisites": ["function_calling"],
                    "module": "建议"
                },
                {
                    "id": "rag_agent_prototype",
                    "label": "RAG Agent 原型",
                    "x": 950,
                    "y": 300,
                    "link": "/tracker",
                    "criteria": "已打通学习记录 + 检索 + 问答的可运行原型，并能持续迭代。",
                    "keywords": ["原型", "rag agent", "闭环", "tracker"],
                    "prerequisites": ["langgraph", "rag"],
                    "module": "评估"
                },
                {
                    "id": "graphrag",
                    "label": "GraphRAG",
                    "x": 1200,
                    "y": 150,
                    "criteria": "可基于实体关系进行可解释检索。",
                    "keywords": ["graphrag", "知识图谱", "关系检索"],
                    "prerequisites": ["rag_agent_prototype"],
                    "module": "评估"
                },
                {
                    "id": "fastapi_deploy",
                    "label": "FastAPI 部署",
                    "x": 1200,
                    "y": 300,
                    "criteria": "可把 Agent 封装成稳定 API 服务并部署。",
                    "keywords": ["fastapi", "部署", "服务化", "接口"],
                    "prerequisites": ["rag_agent_prototype"],
                    "module": "评估"
                }
            ],
            "edges": [
                {"from": "llm_api", "to": "multi_turn"},
                {"from": "multi_turn", "to": "function_calling"},
                {"from": "function_calling", "to": "langgraph"},
                {"from": "function_calling", "to": "rag"},
                {"from": "langgraph", "to": "rag_agent_prototype"},
                {"from": "rag", "to": "rag_agent_prototype"},
                {"from": "rag_agent_prototype", "to": "graphrag"},
                {"from": "rag_agent_prototype", "to": "fastapi_deploy"}
            ]
        }

    def _load_skill_tree(self):
        if os.path.exists(self.skill_tree_file):
            with open(self.skill_tree_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        data = self._default_skill_tree()
        with open(self.skill_tree_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    def _load_or_create_db(self):
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            print("加载已有数据库...")
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            print("初始化新数据库...")
            self.index = None
            self.metadata = []

    def _load_skill_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def _save_skill_progress(self):
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.skill_progress, f, ensure_ascii=False, indent=2)

    def _find_skill_node(self, node_id):
        for node in self.skill_tree.get("nodes", []):
            if node.get("id") == node_id:
                return node
        return None

    def _infer_skill_hits(self, node):
        keywords = [k.lower() for k in node.get("keywords", [])]
        node_label = (node.get("label") or "").lower()
        node_id = node.get("id")
        hit_count = 0
        for record in self.metadata:
            record_skill = (record.get("skill_id") or "").strip()
            if record_skill and record_skill == node_id:
                hit_count += 1
                continue
            text = f"{record.get('title', '')} {record.get('content', '')}".lower()
            if node_label and node_label in text:
                hit_count += 1
                continue
            if any(keyword in text for keyword in keywords):
                hit_count += 1
        return hit_count

    def get_skill_tree_view(self):
        nodes = self.skill_tree.get("nodes", [])
        progress_map = {}
        completed_ids = set()
        for node in nodes:
            node_id = node.get("id")
            stored = self.skill_progress.get(node_id, {})
            stored_score = float(stored.get("confidence", 0))
            hit_count = self._infer_skill_hits(node)
            inferred_score = min(95.0, hit_count * 22.0)
            confidence = round(max(stored_score, inferred_score), 1)
            progress_map[node_id] = {
                "confidence": confidence,
                "evidence_count": hit_count,
                "last_assessed_at": stored.get("last_assessed_at", "")
            }
            if confidence >= 80:
                completed_ids.add(node_id)

        rendered_nodes = []
        first_unlock_set = False
        for node in nodes:
            node_id = node.get("id")
            prereq = node.get("prerequisites", [])
            unlocked = all(p in completed_ids for p in prereq)
            confidence = progress_map[node_id]["confidence"]
            if confidence >= 80:
                status = "completed"
            elif not unlocked:
                status = "locked"
            else:
                if not first_unlock_set:
                    status = "active"
                    first_unlock_set = True
                else:
                    status = "active" if confidence >= 25 else "locked"
            if confidence >= 25 and status == "locked" and unlocked:
                status = "active"
            rendered = dict(node)
            rendered["status"] = status
            rendered["confidence"] = confidence
            rendered["evidence_count"] = progress_map[node_id]["evidence_count"]
            rendered["last_assessed_at"] = progress_map[node_id]["last_assessed_at"]
            rendered_nodes.append(rendered)

        return {
            "track_id": self.skill_tree.get("track_id"),
            "name": self.skill_tree.get("name"),
            "nodes": rendered_nodes,
            "edges": self.skill_tree.get("edges", [])
        }

    def mark_skill_completed(self, node_id):
        node = self._find_skill_node(node_id)
        if not node:
            return False
        self.skill_progress[node_id] = {
            "confidence": 100,
            "status": "completed",
            "last_assessed_at": datetime.now().isoformat(timespec="seconds")
        }
        self._save_skill_progress()
        return True

    def _tokenize(self, text):
        if not text:
            return []
        lower_text = text.lower()
        word_tokens = re.findall(r"[a-z0-9_]+", lower_text)
        cjk_chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
        cjk_bigrams = [a + b for a, b in zip(cjk_chars, cjk_chars[1:])]
        return word_tokens + cjk_chars + cjk_bigrams

    def _bm25_rank(self, query, top_k=3, k1=1.5, b=0.75, candidate_indices=None):
        if not self.metadata:
            return []
        indices = candidate_indices if candidate_indices is not None else list(range(len(self.metadata)))
        if not indices:
            return []
        corpus_tokens = {}
        doc_freq = Counter()
        doc_lengths = {}
        for idx in indices:
            record = self.metadata[idx]
            text = f"{record.get('title', '')} {record.get('content', '')}"
            tokens = self._tokenize(text)
            corpus_tokens[idx] = tokens
            doc_lengths[idx] = len(tokens)
            for token in set(tokens):
                doc_freq[token] += 1
        avgdl = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0.0
        query_tokens = self._tokenize(query)
        if not query_tokens or avgdl == 0:
            return []
        scores = []
        n_docs = len(indices)
        for idx in indices:
            tokens = corpus_tokens[idx]
            tf = Counter(tokens)
            doc_len = doc_lengths[idx]
            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                df = doc_freq.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                freq = tf[token]
                denom = freq + k1 * (1 - b + b * doc_len / avgdl)
                score += idf * (freq * (k1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:top_k]]

    def _vector_rank(self, query, top_k=3, candidate_indices=None):
        if self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array([query_embedding]).astype("float32")
        if candidate_indices is not None:
            if not candidate_indices:
                return []
            query_vec = query_embedding[0]
            scored = []
            for idx in candidate_indices:
                if idx < 0 or idx >= len(self.metadata):
                    continue
                vec = self.index.reconstruct(int(idx))
                dist = float(np.sum((vec - query_vec) ** 2))
                scored.append((idx, dist))
            scored.sort(key=lambda x: x[1])
            return [idx for idx, _ in scored[:top_k]]
        _, indices = self.index.search(query_embedding, top_k)
        valid_indices = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                valid_indices.append(idx)
        return valid_indices

    def _get_collection_indices(self):
        collection_a = []
        collection_b = []
        for idx, record in enumerate(self.metadata):
            source = (record.get("source") or "").strip().lower()
            if source in {"pdf", "upload", "upload_pdf"}:
                collection_b.append(idx)
            else:
                collection_a.append(idx)
        return collection_a, collection_b

    def _hybrid_search_indices(self, query, candidate_indices, top_k=10):
        if not candidate_indices:
            return []
        vector_rank = self._vector_rank(
            query,
            top_k=min(len(candidate_indices), top_k * 3),
            candidate_indices=candidate_indices
        )
        bm25_rank = self._bm25_rank(
            query,
            top_k=min(len(candidate_indices), top_k * 3),
            candidate_indices=candidate_indices
        )
        return self._rrf_fuse([vector_rank, bm25_rank], top_k=min(top_k, len(candidate_indices)), k=60)

    def _rrf_fuse(self, rank_lists, top_k=3, k=60):
        fused = {}
        for rank_list in rank_lists:
            for rank, idx in enumerate(rank_list, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
        final_rank = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in final_rank[:top_k]]

    def _format_text_for_embedding(self, record):
        return f"日期:{record['date']} 主题:{record['title']} 内容:{record['content']}"

    def _build_record_embedding(self, record):
        # 目的：先对内容做 chunking，为长文本接入预留语义切分能力。
        chunks = self.text_splitter.split_text(record['content'])
        if not chunks:
            chunks = [record['content']]
        # 目的：将同一条记录的多个 chunk 分别向量化，保留局部语义。
        chunk_texts = [f"日期:{record['date']} 主题:{record['title']} 内容:{chunk}" for chunk in chunks]
        chunk_embeddings = self.embeddings.embed_documents(chunk_texts)
        chunk_embeddings = np.array(chunk_embeddings).astype("float32")
        # 目的：将 chunk 向量聚合为单条记录向量，兼容当前 metadata 与删除逻辑。
        return chunk_embeddings.mean(axis=0)

    def add(self, records):
        if isinstance(records, dict):
            records = [records]
        if not records:
            return

        print(f"生成 {len(records)} 条 embedding（API）...")
        normalized = []
        for record in records:
            item = dict(record)
            if "source" not in item:
                item["source"] = "manual"
            normalized.append(item)
        embeddings = np.array([self._build_record_embedding(r) for r in normalized]).astype("float32")

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)
        self.metadata.extend(normalized)

        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print("添加完成\n")
        
    def add_document(self, title, content, source="upload", skill_id=None):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        chunks = splitter.split_text(content)
        
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "date": "uploaded",
                "title": f"{title} [chunk {i+1}/{len(chunks)}]",
                "content": chunk,
                "source": source,
                "skill_id": skill_id or ""
            })
        
        self.add(records)
        return len(chunks)
    
    def delete(self, index_to_remove):
        if 0 <= index_to_remove < len(self.metadata):
            self.metadata.pop(index_to_remove)
            if not self.metadata:
                self.index = None
            else:
                embeddings = np.array([self._build_record_embedding(r) for r in self.metadata]).astype("float32")
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings)

            if self.index:
                faiss.write_index(self.index, self.index_file)
            elif os.path.exists(self.index_file):
                os.remove(self.index_file)

            with open(self.meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            return True
        return False

    def hybrid_search(self, query, top_k=10):
        if not self.metadata:
            return []
        final_indices = self._hybrid_search_indices(query, list(range(len(self.metadata))), top_k=top_k)
        return [self.metadata[idx] for idx in final_indices]

    def search(self, query, top_k=3):
        if not self.metadata:
            return []
        collection_a_indices, collection_b_indices = self._get_collection_indices()
        recall_k = max(top_k * 2, 5)
        recalled_a = self._hybrid_search_indices(query, collection_a_indices, top_k=recall_k)
        recalled_b = self._hybrid_search_indices(query, collection_b_indices, top_k=recall_k)
        merged_indices = []
        for idx in recalled_a + recalled_b:
            if idx not in merged_indices:
                merged_indices.append(idx)
        if not merged_indices:
            return []
        candidates = [self.metadata[idx] for idx in merged_indices]
        pairs = [[query, r["content"]] for r in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        return [r for _, r in ranked[:top_k]]

    def evaluate_search_precision_at_3(self):
        case_map = self.skill_tree.get("retrieval_eval_cases", [])
        eval_cases = [(x.get("query", ""), x.get("expected_title", "")) for x in case_map if x.get("query") and x.get("expected_title")]
        if not eval_cases:
            eval_cases = [
                ("我该怎么定位Agent方向的职业目标？", "市场锚定"),
                ("我需要补强简历表达能力，应该回顾什么？", "简历分析"),
                ("怎么理解多轮对话的上下文管理？", "接口实战"),
                ("LangGraph 的核心编排思想是什么？", "架构初探"),
                ("RAG 怎么降低模型胡说八道？", "数据增强")
            ]
        hit_count = 0
        for query, expected_title in eval_cases:
            results = self.search(query, top_k=3)
            recalled_titles = [r["title"] for r in results]
            hit = expected_title in recalled_titles
            hit_count += int(hit)
            print(f"[Eval] Query: {query}")
            print(f"[Eval] 期望: {expected_title} | 召回: {recalled_titles} | Hit: {hit}\n")
        precision_at_3 = hit_count / (len(eval_cases) * 3)
        print(f"[Eval] Precision@3 = {precision_at_3:.4f} ({hit_count}/{len(eval_cases) * 3})")
        return {"precision_at_3": precision_at_3, "hits": hit_count, "queries": len(eval_cases), "k": 3}

    def _records_to_context(self, results):
        if not results:
            return "无相关知识"
        return "\n".join([f"[{r['date']}] {r['title']} - {r['content']}" for r in results])

    def ask(self, query, top_k=3):
        results = self.search(query, top_k)
        context = self._records_to_context(results)

        prompt = f"""
你是一个个人知识助手，请基于以下记录回答问题：

【历史记录】
{context}

【问题】
{query}

要求：
1. 优先基于记录回答
2. 如果信息不足，可以补充
"""
        response = self.llm.invoke(prompt)
        return response.content

    def consult(self, query, node_id=None, top_k=5):
        node = self._find_skill_node(node_id) if node_id else None
        scope = node.get("label") if node else "Agent 学习路线"
        search_query = f"{query} {scope}"
        results = self.search(search_query, top_k=top_k)
        context = self._records_to_context(results)
        prompt = f"""
你是学习咨询助手。你需要根据历史记录和当前技能路线，输出学习诊断。

【咨询问题】
{query}

【技能范围】
{scope}

【历史记录】
{context}

输出要求：
1. 先判断当前阶段（起步/进阶/冲刺）
2. 给出 3 条关键问题诊断
3. 给出 1 个最优先改进行动
"""
        response = self.llm.invoke(prompt)
        return {
            "response": response.content if hasattr(response, "content") else str(response),
            "references": results
        }

    def advise(self, node_id, top_k=5):
        node = self._find_skill_node(node_id)
        if not node:
            return {"response": "未找到对应技能节点。", "references": []}
        label = node.get("label", node_id)
        criteria = node.get("criteria", "")
        results = self.search(f"{label} 学习记录 实践 复盘", top_k=top_k)
        context = self._records_to_context(results)
        prompt = f"""
你是学习路径建议助手，需要给出可执行学习计划。

【目标节点】
{label}

【完成标准】
{criteria}

【学习记录】
{context}

请输出：
1. 当前差距
2. 接下来 3 个具体任务（每个任务包含产出物）
3. 一个自测问题
"""
        response = self.llm.invoke(prompt)
        return {
            "response": response.content if hasattr(response, "content") else str(response),
            "references": results
        }

    def evaluate_node(self, node_id, history=None, user_answer=""):
        node = self._find_skill_node(node_id)
        if not node:
            return {"response": "未找到对应技能节点。", "passed": False, "scores": {"rule": 0, "llm": 0, "final": 0}}
        history = history or []
        node_name = node.get("label", node_id)
        criteria = node.get("criteria", "")
        context_records = self.search(f"{node_name} 学习记录 评估", top_k=5)
        context = self._records_to_context(context_records)

        history_lines = []
        if isinstance(history, list):
            for item in history:
                role = (item.get("role") or "").strip()
                content = (item.get("content") or "").strip()
                if role and content:
                    history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "（无）"
        answer_text = user_answer.strip() if user_answer else "（首轮，请直接出一道核心概念题）"
        evidence_count = self._infer_skill_hits(node)
        rule_score = min(100, 35 + evidence_count * 10 + (10 if len(answer_text) > 40 else 0))
        prompt = f"""
你是严格但友善的技术评估官，正在评估用户对"{node_name}"的掌握程度。

【完成标准】
{criteria}

【学习记录】
{context}

【已有对话历史】
{history_text}

【用户本轮回答】
{answer_text}

请输出 JSON，格式如下：
{{
  "message": "你这一轮对用户说的话",
  "llm_score": 0-100 的整数,
  "passed": true 或 false
}}

评估规则：
1. 第一次调用时，优先出题
2. 回答到位就追问更深，不到位就指出缺失并追问
3. 当可以结束时，message 中必须包含 [基本掌握] 或 [建议继续学习]
"""
        response = self.llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        llm_score = 0
        passed = False
        message = raw
        try:
            data = json.loads(raw)
            message = str(data.get("message", "")).strip() or raw
            llm_score = int(data.get("llm_score", 0))
            passed = bool(data.get("passed", False))
        except Exception:
            score_match = re.search(r'"?llm_score"?\s*[:：]\s*(\d+)', raw)
            if score_match:
                llm_score = int(score_match.group(1))
            if "[基本掌握]" in raw:
                passed = True
        if "[基本掌握]" in message:
            passed = True
        if "[建议继续学习]" in message:
            passed = False
        llm_score = max(0, min(100, llm_score))
        final_score = round(rule_score * 0.4 + llm_score * 0.6, 1)
        if passed:
            stored = self.skill_progress.get(node_id, {})
            self.skill_progress[node_id] = {
                "confidence": max(85, final_score),
                "status": "completed",
                "last_assessed_at": datetime.now().isoformat(timespec="seconds"),
                "last_scores": {"rule": rule_score, "llm": llm_score, "final": final_score},
                "attempts": int(stored.get("attempts", 0)) + 1
            }
            self._save_skill_progress()
        else:
            stored = self.skill_progress.get(node_id, {})
            self.skill_progress[node_id] = {
                "confidence": max(float(stored.get("confidence", 0)), min(75, final_score)),
                "status": "learning",
                "last_assessed_at": datetime.now().isoformat(timespec="seconds"),
                "last_scores": {"rule": rule_score, "llm": llm_score, "final": final_score},
                "attempts": int(stored.get("attempts", 0)) + 1
            }
            self._save_skill_progress()
        return {
            "response": message,
            "passed": passed,
            "scores": {
                "rule": rule_score,
                "llm": llm_score,
                "final": final_score
            }
        }


if __name__ == "__main__":
    agent = PersonalBrainAgent()
    if agent.index is None or agent.index.ntotal == 0:
        agent.add([
            {"date": "3/14", "title": "市场锚定", "content": "通过招聘端调研，确立了以“Agent 应用开发”为核心的职业选择，从盲目学到为需而学。", "skill_id": "llm_api"},
            {"date": "3/15", "title": "简历分析", "content": "将过往经验整理，通过简历修改完成了对自身技能树需求发展的了解", "skill_id": "llm_api"},
            {"date": "3/15", "title": "行业情报搜集", "content": "参加大型招聘会，实地了解企业对 Agent 开发的需要程度", "skill_id": "llm_api"},
            {"date": "3/16", "title": "接口实战", "content": "突破 LLM API 调用关卡，了解多轮对话中的上下文管理", "skill_id": "multi_turn"},
            {"date": "3/16", "title": "行动力进化", "content": "成功实现 Function Calling，让AI在聊天过程中监控用户的情绪变化从而改变回答的质地。", "skill_id": "function_calling"},
            {"date": "3/16", "title": "架构初探", "content": "接触 LangGraph，理解了基于状态机（State Machine）的 Agent 编排逻辑。", "skill_id": "langgraph"},
            {"date": "3/16", "title": "数据增强", "content": "浅层实现 RAG（检索增强生成），解决了模型“胡说八道”的问题，实现了外部知识的闭环。", "skill_id": "rag"},
            {"date": "3/16", "title": "原型闭环", "content": "完成了首个“RAG Agent”原型，跑通了“感知-检索-思考-执行”的完整链路。", "skill_id": "rag_agent_prototype"},
            {"date": "3/17", "title": "产品化思维", "content": "从写代码转向写 PRD，开始考虑“个人思考记录”项目的用户场景与逻辑细节。", "skill_id": "rag_agent_prototype"},
            {"date": "3/17", "title": "深度攻坚", "content": "尝试将复杂的规划逻辑融入实际应用，尽可能从 Demo 迈向真正的产品。", "skill_id": "rag_agent_prototype"}
        ])

    print("=== 提问 ===")
    answer = agent.ask("我下一步应该学什么?")
    print(answer)
    print("\n=== 检索评估 ===")
    agent.evaluate_search_precision_at_3()
