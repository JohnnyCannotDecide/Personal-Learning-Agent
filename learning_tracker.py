import os
import json
import faiss
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class PersonalBrainAgent:
    def __init__(self, db_dir="learning_db"):
        self.db_dir = db_dir
        self.index_file = os.path.join(db_dir, "vector_index.faiss")
        self.meta_file = os.path.join(db_dir, "metadata.json")

        os.makedirs(self.db_dir, exist_ok=True)
        # ✅ 你指定的 API 形式
        OPENROUTER_API_KEY = ""
        self.llm = ChatOpenAI(
            model="openrouter/hunter-alpha",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
        )

        self.embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-8B",
            base_url="https://api.siliconflow.com/v1",
            api_key=""
        )

        print("已加载 API 模型（LLM + Embedding）")

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

    def _format_text_for_embedding(self, record):
        return f"日期:{record['date']} 主题:{record['title']} 内容:{record['content']}"

    def add(self, records):
        if isinstance(records, dict):
            records = [records]

        texts = [self._format_text_for_embedding(r) for r in records]

        print(f"生成 {len(records)} 条 embedding（API）...")

        # ✅ LangChain embedding
        embeddings = self.embeddings.embed_documents(texts)
        embeddings = np.array(embeddings).astype("float32")

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

    def delete(self, index_to_remove):
            if 0 <= index_to_remove < len(self.metadata):
                # 1. 从 metadata 移除
                self.metadata.pop(index_to_remove)
                
                # 2. 重新构建 FAISS 索引
                if not self.metadata:
                    self.index = None
                else:
                    texts = [self._format_text_for_embedding(r) for r in self.metadata]
                    embeddings = self.embeddings.embed_documents(texts)
                    embeddings = np.array(embeddings).astype("float32")
                    
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
    def search(self, query, top_k=3):
        if self.index is None or self.index.ntotal == 0:
            return []

        # ✅ 查询 embedding
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array([query_embedding]).astype("float32")

        D, I = self.index.search(query_embedding, top_k)

        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results

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
