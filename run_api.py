"""
Process entrypoint for the API server.

Exists for one reason: on Windows, psycopg refuses to run on asyncio's
ProactorEventLoop ("Psycopg cannot use the 'ProactorEventLoop' to run in async
mode"). AsyncPostgresSaver therefore raised at every startup and the
orchestrator degraded to MemorySaver — durable checkpointing was silently dead
on every Windows run, and the only trace was one WARNING line.

Setting asyncio's event loop policy does not fix this, which is the part worth
knowing. uvicorn 0.49 does not consult the policy: `uvicorn/loops/asyncio.py`
returns `asyncio.ProactorEventLoop` as an explicit loop *factory* on win32, and
`Server.run()` passes that factory straight to the runner. A factory overrides
the policy, so `set_event_loop_policy(WindowsSelectorEventLoopPolicy())` is inert
no matter how early it runs.

So the loop is constructed here instead, and uvicorn is handed a server to run
on it rather than being asked to make its own.

    python run_api.py

`uvicorn api.main:app` remains correct on Linux, where the default loop is
already selector-based — the Docker image is unaffected and keeps uvloop. On
Windows that command still degrades to in-memory checkpointing.

The selector loop's limits are real but do not bind here: ~512 sockets and no
subprocess support. This process makes HTTP calls to Qdrant and Postgres and
spawns nothing.
"""

import asyncio
import os
import sys

import uvicorn


def main() -> None:
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))

    if sys.platform != "win32":
        # Let uvicorn pick its own loop so uvloop is still used when installed.
        uvicorn.run("api.main:app", host=host, port=port, workers=1)
        return

    config = uvicorn.Config("api.main:app", host=host, port=port, workers=1)
    server = uvicorn.Server(config)
    # asyncio.Runner is 3.11+; loop_factory is the only way to defeat uvicorn's
    # hardcoded Proactor factory without monkeypatching its internals.
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(server.serve())


if __name__ == "__main__":
    main()
