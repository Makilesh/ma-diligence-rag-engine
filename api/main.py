"""
FastAPI application entry point with lifespan.

Manages startup/shutdown of Qdrant client, BudgetTracker, and LangGraph.
Immutable audit log on every query.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.vector_db.qdrant_client import get_qdrant_client, close_qdrant_client
from src.vector_db.collection_manager import setup_collections
from src.llm.budget_tracker import BudgetTracker
from src.workflow.orchestrator import get_compiled_graph, close_checkpointer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Windows defaults asyncio to ProactorEventLoop, which psycopg refuses to run
# on, so AsyncPostgresSaver raises at startup and checkpointing degrades to
# MemorySaver. See run_api.py for the full account.
#
# This covers callers that import the app before any loop exists (tests,
# scripts, `python -c`). It does NOT cover `uvicorn api.main:app`, because
# uvicorn imports the app from inside Server.serve(), by which point its loop is
# already running and a policy change has no effect — that path needs
# run_api.py. Setting it in both places is deliberate: neither one covers the
# other's entry.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Module-level reference to compiled graph
_app_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler — startup and shutdown logic.
    """
    global _app_graph

    logger.info("Starting M&A Due Diligence Intelligence Engine")

    # Initialize Qdrant
    client = get_qdrant_client()
    await setup_collections(client)

    # Initialize budget tracker
    postgres_url = os.getenv(
        "POSTGRES_URL",
        "postgresql://manda_user:password@localhost:5432/manda_rag",
    )
    await BudgetTracker.get_instance(postgres_url)

    # Compile LangGraph
    _app_graph = await get_compiled_graph(postgres_url)

    logger.info("Application startup complete")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down application")
    await close_checkpointer()
    await close_qdrant_client()
    await BudgetTracker.close()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="M&A Due Diligence Intelligence Engine",
    description="Production-grade Hybrid Agentic RAG for M&A Due Diligence",
    version="1.0.0",
    lifespan=lifespan,
)


def get_graph():
    """Returns the compiled LangGraph application."""
    return _app_graph


# Register routes
from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router
from api.routes.deals import router as deals_router

app.include_router(query_router, prefix="/api/v1", tags=["queries"])
app.include_router(ingest_router, prefix="/api/v1", tags=["ingestion"])
app.include_router(deals_router, prefix="/api/v1", tags=["deals"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "manda-rag"}
