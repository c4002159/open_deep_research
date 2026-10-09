"""Run the service:  python -m open_deep_research.service  (env: ODR_REDIS_URL, ODR_API_KEYS, ...)"""

import asyncio
import os
import sys

import uvicorn

from open_deep_research.service.app import create_app

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # psycopg async needs it
    uvicorn.run(create_app(), host=os.environ.get("ODR_HOST", "127.0.0.1"), port=int(os.environ.get("ODR_PORT", "8080")))
