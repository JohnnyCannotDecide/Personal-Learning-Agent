from learning_tracker import PersonalBrainAgent

agent = PersonalBrainAgent()

test_cases = [
    {"query": "LangGraph是什么", "expected": "架构初探"},
    {"query": "RAG怎么实现", "expected": "数据增强"},
    {"query": "我学了哪些API相关内容", "expected": "接口实战"},
    # 再加两条，其中一条用你刚才上传的资料里的内容来构造 query
    # 比如你上传了一篇讲 embedding 的文章，就加一条和 embedding 相关的 query
]

correct = 0
for case in test_cases:
    results = agent.search(case["query"], top_k=3)
    titles = [r["title"] for r in results]
    hit = case["expected"] in titles
    print(f"Query: {case['query']}")
    print(f"召回: {titles}")
    print(f"命中: {'✓' if hit else '✗'}\n")
    if hit:
        correct += 1

print(f"Precision@3: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.0f}%")