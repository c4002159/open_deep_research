"""Show what the agent is actually told to research for one eval case (2 LLM calls).

    python -m eval.inspect_brief kb_phone_mask

The researcher can only be as focused as its brief. This prints (1) the research brief the
planner writes from the user's question and (2) the topic the supervisor delegates to a researcher.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from eval.cases import ALL_CASES


async def main(case_id: str) -> None:
    load_dotenv()
    from open_deep_research import deep_researcher as dr

    case = next(c for c in ALL_CASES if c["id"] == case_id)
    cfg = {"configurable": {"allow_clarification": False}}
    state = {"messages": [HumanMessage(content=case["question"])]}

    cmd = await dr.write_research_brief(state, cfg)
    brief = cmd.update["research_brief"]
    print("USER QUESTION:\n ", case["question"], "\n")
    print(f"RESEARCH BRIEF ({len(brief)} chars):\n{brief}\n")

    sup_state = {"supervisor_messages": cmd.update["supervisor_messages"]["value"], "research_brief": brief, "research_iterations": 0}
    cmd2 = await dr.supervisor(sup_state, cfg)
    msg = cmd2.update["supervisor_messages"][0]
    for tc in msg.tool_calls:
        print(f"SUPERVISOR -> {tc['name']}:")
        for k, v in tc["args"].items():
            print(f"   {k}: {v}\n")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(sys.argv[1]))
