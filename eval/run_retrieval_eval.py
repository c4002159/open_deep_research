"""用DeepEval官方Contextual Recall/Precision指标评测RAG检索质量。"""
import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

from open_deep_research.utils import _get_kb_vectorstore
from eval.retrieval_gold_set import RETRIEVAL_GOLD_CASES
from eval.deepseek_judge import DeepSeekJudge


K = 3  # 跟 search_knowledge_base 工具里用的 k 值保持一致
THRESHOLD = 0.7  # 业界常见目标是 > 0.8，这里先用 0.7 起步，看实际分数再调


async def build_test_case(case: dict, config: dict) -> LLMTestCase:
    vectorstore = _get_kb_vectorstore(config)
    results = await vectorstore.asimilarity_search(case["query"], k=K)
    retrieved_chunks = [doc.page_content for doc in results]

    return LLMTestCase(
        input=case["query"],
        actual_output="N/A",  # 这两个指标只看检索质量，不评价生成的最终答案
        expected_output=case["gold_answer"],
        retrieval_context=retrieved_chunks,
    )


async def main():
    config = {"configurable": {}}  # postgres_connection_string 用Configuration默认值
    test_cases = []
    for case in RETRIEVAL_GOLD_CASES:
        print(f"▶ 正在构造测试用例: {case['id']} ...")
        test_case = await build_test_case(case, config)
        test_cases.append(test_case)

    judge = DeepSeekJudge()
    evaluate(
        test_cases,
        [
            ContextualRecallMetric(threshold=THRESHOLD, model=judge),
            ContextualPrecisionMetric(threshold=THRESHOLD, model=judge),
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())