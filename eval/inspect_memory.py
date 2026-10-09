"""List what the agent has stored in long-term memory (mem0) for a user - READ ONLY.

    python -m eval.inspect_memory [user_id]

The agent saves "durable facts" between runs. A wrong fact saved once is reused by every later run,
so a one-off wrong answer should be checked against this list.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from mem0 import MemoryClient


def main(user_id: str) -> None:
    load_dotenv()
    client = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
    res = client.get_all(filters={"user_id": user_id})
    mems = res.get("results", res) if isinstance(res, dict) else res
    print(f"{len(mems)} stored memories for user_id={user_id!r}\n")
    for m in sorted(mems, key=lambda m: m.get("created_at", "")):
        print(f"- {m.get('created_at', '')[:19]} | {m['memory'][:260]}".replace("\n", " "))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "default_user")
