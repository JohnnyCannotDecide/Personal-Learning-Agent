"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
# 1 定义 State
class ChatState(TypedDict):
    user_input: str
    reply: str
# 2 初始化 LLM
llm = ChatOpenAI(
    model="openrouter/hunter-alpha",
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)
# 3 节点1：接收输入
def input_node(state: ChatState):
    text = input("用户: ")
    return {
        "user_input": text,
        "retry_count": 0
    }
# 4 节点2：LLM生成
def llm_node(state: ChatState):
    response = llm.invoke(state["user_input"])
    return {
        "reply": response.content,
        "retry_count": state.get("retry_count", 0) + 1
    }
# 5 节点3：质量评估
def evaluation_node(state: ChatState):
    reply = state["reply"]
    print("\n[评估节点] 当前回复长度:", len(reply))
    return {}

# 6 条件判断函数
def check_quality(state: ChatState):
    reply = state["reply"]
    retry = state.get("retry_count", 0)
    if len(reply) < 50 and retry < 3:
        print("[评估结果] 回复太短，重新生成...")
        return "retry"
    else:
        print("[评估结果] 回复通过或达到最大重试次数")
        return "end"

# 7 构建Graph
builder = StateGraph(ChatState)
builder.add_node("input", input_node)
builder.add_node("llm", llm_node)
builder.add_node("evaluate", evaluation_node)
builder.set_entry_point("input")
builder.add_edge("input", "llm")
builder.add_edge("llm", "evaluate")
# 条件边
builder.add_conditional_edges(
    "evaluate",
    check_quality,
    {
        "retry": "llm",
        "end": END
    }
)
graph = builder.compile()

# 8 运行
result = graph.invoke({})
print("\nAI:", result["reply"])

"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# 1 原始文本、State
text = """
Agent是一种能够感知环境、进行推理并采取行动的AI系统。与普通LLM不同，Agent可以调用外部工具，比如搜索引擎、代码执行器、数据库等。Agent的核心循环是：感知输入，推理下一步行动，执行工具，观察结果，再次推理，直到任务完成。LangGraph是一个用图结构来管理Agent状态和流程的框架，它允许Agent有分支、循环和多个节点协同工作。RAG是检索增强生成，它把外部知识库切成小块转成向量存储，每次提问时先检索相关片段再交给LLM回答，解决了上下文窗口有限的问题。Function Calling允许LLM自己决定什么时候调用什么工具，而不是由人用if语句硬编码控制流程。Multi-Agent架构是多个Agent协同工作，由一个Orchestrator负责调度，每个子Agent负责专门的任务。
"""
class GraphState(TypedDict):
    question: str
    context: str
    score: float
    answer: str
# 2 文本切块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=20
)
docs = splitter.create_documents([text])
    
# 3 Embedding 模型
embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    base_url="https://api.siliconflow.com/v1",
    api_key=""
)

# 4 构建 FAISS 向量数据库
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# 5 LLM
OPENROUTER_API_KEY = ""
llm = ChatOpenAI(
    model="openrouter/hunter-alpha",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

# 6 检索节点
def retrieve(state: GraphState):
    question = state["question"]
    results = vectorstore.similarity_search_with_score(question, k=1)
    doc, score = results[0]
    return {
        "context": doc.page_content,
        "score": score
    }
# 7 RAG回答
def rag_answer(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """
根据以下知识库内容回答问题。
如果无法从知识库得到答案，请回答不知道。

知识库:
{context}
问题:
{question}
"""
    )
    messages = prompt.format_messages(
        context=state["context"],
        question=state["question"]
    )
    response = llm.invoke(messages)
    return {"answer": response.content}
# 8 LLM自由回答
def llm_answer(state: GraphState):

    prompt = f"""
知识库中没有找到相关内容，以下是我自己的判断：

问题: {state["question"]}
"""
    response = llm.invoke(prompt)
    return {"answer": response.content}
# 9 路由判断
def route(state: GraphState):
    if state["score"] < 0.7:
        return "llm"
    return "rag"
# 10 构建 LangGraph
builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve)
builder.add_node("rag", rag_answer)
builder.add_node("llm", llm_answer)

builder.set_entry_point("retrieve")

builder.add_conditional_edges(
    "retrieve",
    route,
    {
        "rag": "rag",
        "llm": "llm"
    }
)
builder.add_edge("rag", END)
builder.add_edge("llm", END)

graph = builder.compile()
# 11 运行
result = graph.invoke({
    "question": "LangGraph是什么？"
})
print("\n最终回答:\n")
print(result["answer"])
