import os
import json
import faiss
import numpy as np
import math
import re
from collections import Counter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

class PersonalBrainAgent:
    def __init__(self, db_dir="learning_db"):
        self.db_dir = db_dir
        self.index_file = os.path.join(db_dir, "vector_index.faiss")
        self.meta_file = os.path.join(db_dir, "metadata.json")

        os.makedirs(self.db_dir, exist_ok=True)
        # ✅ 你指定的 API 形式
        load_dotenv()

        self.llm = ChatOpenAI(
            model="google/gemma-3-12b-it:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
        )

        self.embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-8B",
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
        )
        # 目的：引入可复用的 chunking 流程，短文本几乎不切分，长文本自动切分。
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        print("已加载 API 模型（LLM + Embedding）")

        self.reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)

        self._load_or_create_db()

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
        # 目的：每条记录先执行 chunking，再聚合成单向量写入 FAISS。
        embeddings = np.array([self._build_record_embedding(r) for r in records]).astype("float32")

        # 初始化 FAISS
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)
        self.metadata.extend(records)

        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print("添加完成\n")
        
    def add_document(self, title, content, source="upload"):
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
                "source": source  # 区分是上传资料还是学习记录
            })
        
        self.add(records)
        return len(chunks)  # 返回切了多少块，方便前端显示
    
    def delete(self, index_to_remove):
            if 0 <= index_to_remove < len(self.metadata):
                # 1. 从 metadata 移除
                self.metadata.pop(index_to_remove)
                
                # 2. 重新构建 FAISS 索引
                if not self.metadata:
                    self.index = None
                else:
                    # 目的：删除后按同一套 chunking + 聚合逻辑重建索引，保持检索一致性。
                    embeddings = np.array([self._build_record_embedding(r) for r in self.metadata]).astype("float32")
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    self.index.add(embeddings)
                
                # 3. 持久化
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
        # 目的：构造可复现的检索评估集，把“感觉准确”变成可验证数字。
        eval_cases = [
            ("我该怎么定位Agent方向的职业目标？", "市场锚定"),
            ("我需要补强简历表达能力，应该回顾什么？", "简历分析"),
            ("怎么理解多轮对话的上下文管理？", "接口实战"),
            ("LangGraph 的核心编排思想是什么？", "架构初探"),
            ("RAG 怎么降低模型胡说八道？", "数据增强"),
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

    def ask(self, query, top_k=3):
        results = self.search(query, top_k)

        if not results:
            context = "无相关知识"
        else:
            context = "\n".join([
                f"[{r['date']}] {r['title']} - {r['content']}"
                for r in results
            ])

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
# 测试
if __name__ == "__main__":
    agent = PersonalBrainAgent()
    if agent.index is None or agent.index.ntotal == 0:
        agent.add([
            {"date": "3/14", "title": "市场锚定", "content": "通过招聘端调研，确立了以“Agent 应用开发”为核心的职业选择，从盲目学到为需而学。"},
            {"date": "3/15", "title": "简历分析", "content": "将过往经验整理，通过简历修改完成了对自身技能树需求发展的了解"},
            {"date": "3/15", "title": "行业情报搜集", "content": "参加大型招聘会，实地了解企业对 Agent 开发的需要程度"},
            {"date": "3/16", "title": "接口实战", "content": "突破 LLM API 调用关卡，了解多轮对话中的上下文管理"},
            {"date": "3/16", "title": "行动力进化", "content": "成功实现 Function Calling，让AI在聊天过程中监控用户的情绪变化从而改变回答的质地。"},
            {"date": "3/16", "title": "架构初探", "content": "接触 LangGraph，理解了基于状态机（State Machine）的 Agent 编排逻辑。"},
            {"date": "3/16", "title": "数据增强", "content": "浅层实现 RAG（检索增强生成），解决了模型“胡说八道”的问题，实现了外部知识的闭环。"},
            {"date": "3/16", "title": "原型闭环", "content": "完成了首个“RAG Agent”原型，跑通了“感知-检索-思考-执行”的完整链路。"},
            {"date": "3/17", "title": "产品化思维", "content": "从写代码转向写 PRD，开始考虑“个人思考记录”项目的用户场景与逻辑细节。"},
            {"date": "3/17", "title": "深度攻坚", "content": "尝试将复杂的规划逻辑融入实际应用，尽可能从 Demo 迈向真正的产品。"}
        ])

    print("=== 提问 ===")
    answer = agent.ask("我下一步应该学什么?")
    print(answer)
    print("\n=== 检索评估 ===")
    agent.evaluate_search_precision_at_3()
