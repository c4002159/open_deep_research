"""Audit (and optionally clean) the agent's long-term memory against the memory policy.

    python -m eval.clean_memory [user_id]                 # DRY RUN: list keep / drop, change nothing
    python -m eval.clean_memory [user_id] --apply         # delete the entries the policy rejects

Deleting is irreversible and happens in the mem0 cloud account, so --apply must be asked for explicitly.
The agent already ignores rejected entries when reading, so cleaning is housekeeping, not a requirement.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from mem0 import MemoryClient

from open_deep_research.memory_policy import check_fact
from open_deep_research.schema_context import get_schema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("user_id", nargs="?", default="default_user")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    load_dotenv()
    client = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
    res = client.get_all(filters={"user_id": a.user_id})
    mems = res.get("results", res) if isinstance(res, dict) else res

    known = None
    if os.getenv("INTERNAL_DB_URL"):
        try:
            known = get_schema(os.environ["INTERNAL_DB_URL"]).identifiers
        except Exception:
            pass

    keep, drop = [], []
    for m in sorted(mems, key=lambda m: m.get("created_at", "")):
        v = check_fact(m["memory"], known)
        (keep if v.ok else drop).append((m, v))

    print(f"{len(mems)} memories for {a.user_id!r}: keep {len(keep)}, drop {len(drop)}\n")
    print("KEEP:")
    for m, v in keep:
        print(f"  [{v.category}] {m['memory'][:150]}".replace("\n", " "))
    print("\nDROP:")
    for m, v in drop:
        print(f"  ({v.message[:90]}) {m['memory'][:110]}".replace("\n", " "))

    if a.apply and drop:
        for m, _ in drop:
            client.delete(m["id"])
        print(f"\ndeleted {len(drop)} memories")
    elif drop:
        print("\n(dry run - nothing deleted; add --apply to delete the DROP list)")


if __name__ == "__main__":
    main()
