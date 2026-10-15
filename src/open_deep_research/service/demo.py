"""One-command demo:  python -m open_deep_research.service.demo   (needs Redis at ODR_REDIS_URL, default :6390)

Builds data/demo.db from the tracked test.db (with a little refund history), turns the refund workflow on, creates
five demo accounts, and serves the web client at http://127.0.0.1:8080/.  Every default can be overridden by
setting the environment variable first.  ODR_DEMO_MODE exposes the demo tokens at /demo/users: never use this
entry point in production.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

DEMO_DB = Path("data/demo.db")


def demo_env(db_path: Path) -> dict[str, str]:
    from open_deep_research import demo_data as D

    return {
        "ENABLE_ACTIONS": "1",
        "INTERNAL_DB_URL": "sqlite:///" + db_path.resolve().as_posix(),
        "BUSINESS_DATE": D.BUSINESS_DATE,
        "ODR_API_KEYS": ",".join(f"{k}:{u}" for k, u in D.DEMO_KEYS.items()),
        "ODR_ROLES": ",".join(f"{u}={r}" for u, r in D.DEMO_ROLES.items()),
        "ODR_APPROVERS": "bob,carol,dave",
        "ODR_ADMINS": "root",
        "ODR_DEMO_MODE": "1",
        "ODR_DAILY_TOKEN_BUDGET": "400000",
        "ODR_RATE_LIMIT": "20",
        "ODR_MAX_ACTIVE": "4",
    }


def prepare(reset: bool = False, source: str = "test.db") -> Path:
    from open_deep_research import demo_data as D

    if reset or not DEMO_DB.exists():
        D.build_demo_db(str(DEMO_DB), source=source, seed=True)
    for k, v in demo_env(DEMO_DB).items():
        os.environ.setdefault(k, v)
    return DEMO_DB


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reset", action="store_true", help="rebuild data/demo.db (drops all refunds made so far)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("ODR_PORT", "8080")))
    args = ap.parse_args(argv)

    import uvicorn
    from dotenv import load_dotenv

    prepare(reset=args.reset)          # our defaults go in first: load_dotenv never overrides what is already set
    load_dotenv()                      # model names and API keys still come from .env
    from open_deep_research.service.app import create_app

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(f"\n  demo ready ->  http://127.0.0.1:{args.port}/   (accounts: alice, bob, carol, dave, root)\n")
    uvicorn.run(create_app(), host=os.environ.get("ODR_HOST", "127.0.0.1"), port=args.port)


if __name__ == "__main__":
    main()
