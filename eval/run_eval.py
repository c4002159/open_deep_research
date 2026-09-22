from dotenv import load_dotenv
load_dotenv()  
import asyncio
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from open_deep_research.deep_researcher import deep_researcher
from eval.dataset import EVAL_CASES
from eval.metrics import ExecutionAccuracyMetric


async def run_agent(question: str, config: dict) -> str:
    result = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    return result.get("final_report", str(result))


async def main():
    import os  
    test_cases = []
    for case in EVAL_CASES:
        print(f"\n▶ 正在跑: {case['id']} ...")
        config = {"configurable": {"thread_id": f"eval-{case['id']}", "allow_clarification": False}}
        actual_output = await run_agent(case["question"], config)
        print(f"  完成，报告长度: {len(actual_output)} 字符")
        
        # ====== 新增动态判断逻辑 ======
        db_available = bool(os.getenv("INTERNAL_DB_URL"))
        
        if db_available and "expected_output_db_available" in case:
            expected = case["expected_output_db_available"]
            mode = case.get("match_mode_db_available", "all")
        elif not db_available and "expected_output_db_unavailable" in case:
            expected = case["expected_output_db_unavailable"]
            mode = case.get("match_mode_db_unavailable", "any")
        else:
            expected = case["expected_output"]
            mode = case.get("match_mode", "all")
            
        test_cases.append(LLMTestCase(
            input=case["question"],
            actual_output=actual_output,
            expected_output=expected,
            additional_metadata={"match_mode": mode},
        ))

    evaluate(test_cases, [ExecutionAccuracyMetric(threshold=1.0)])


if __name__ == "__main__":
    asyncio.run(main())