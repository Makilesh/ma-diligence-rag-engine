"""
Qdrant client singleton management.

CRITICAL: AsyncQdrantClient must be created inside a running event loop.
This function lazily initializes the client on first call, which will
always happen inside an async context (from a request handler or startup).

The singleton pattern ensures all agents and request handlers share
one connection pool instead of creating per-request clients.
"""

import os
from qdrant_client import AsyncQdrantClient

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_qdrant_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    """
    Returns a singleton AsyncQdrantClient instance.

    CRITICAL: AsyncQdrantClient must be created inside a running event loop.
    This function lazily initializes the client on first call, which will
    always happen inside an async context (from a request handler or startup).

    The singleton pattern ensures all agents and request handlers share
    one connection pool instead of creating per-request clients.

    Configuration via environment variables:
        QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
        QDRANT_GRPC_PORT: gRPC port for faster operations (default: 6334)

    Returns:
        Shared AsyncQdrantClient instance.

    Raises:
        RuntimeError: If called before event loop is available.
    """
    global _qdrant_client
    if _qdrant_client is None:
        from urllib.parse import urlparse
        import socket

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        parsed = urlparse(qdrant_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333

        server_reachable = False
        try:
            with socket.create_connection((host, port), timeout=1.0):
                server_reachable = True
        except OSError:
            pass

        if server_reachable:
            grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
            logger.info(
                "Initializing AsyncQdrantClient singleton (Server Mode)",
                extra={"qdrant_url": qdrant_url, "grpc_port": grpc_port},
            )
            _qdrant_client = AsyncQdrantClient(
                url=qdrant_url,
                grpc_port=grpc_port,
                prefer_grpc=True,  # gRPC is faster for batch operations
                timeout=30,
            )
        else:
            local_path = "./qdrant_local_db"
            logger.warning(
                f"Qdrant server at {qdrant_url} is unreachable. Falling back to local storage mode at {local_path}."
            )
            _qdrant_client = AsyncQdrantClient(path=local_path)
    return _qdrant_client


async def assert_server_compatible(client: AsyncQdrantClient) -> None:
    """
    Fails startup if the Qdrant server is too far from the installed client.

    Qdrant's contract is that major versions match and minor versions differ by
    at most one. The client already notices a violation — and only emits a
    `UserWarning`, which is invisible under a JSON log handler.

    That is far too quiet for what actually happens. Running client 1.18.0
    against server 1.12.1, *every* ingest failed at upsert ("Vector dimension
    error: expected dim: 1024, got 0", "Conversion between sparse and regular
    vectors failed") because the newer client's gRPC vector encoding was
    unreadable by the older server. REST on the same pair worked, so the health
    check passed, collections were created, queries ran, and the engine simply
    answered "I was unable to find sufficient relevant information" 41 times
    against an empty index. Nothing in the pipeline treats an empty index as an
    error, so the only symptom was uniformly bad results.

    A version skew that makes writes silently fail should stop the process, not
    add a line to a log nobody reads.

    Args:
        client: The initialized Qdrant client.

    Raises:
        RuntimeError: If the server version is incompatible with the client.
    """
    try:
        from qdrant_client import version as _qc_version

        client_version = _qc_version.__version__
    except Exception:
        import importlib.metadata

        client_version = importlib.metadata.version("qdrant-client")

    try:
        info = await client.info()
        server_version = info.version
    except Exception as e:
        # Local-storage mode has no server; nothing to be incompatible with.
        logger.info(f"Skipping Qdrant version check: {e}")
        return

    def _major_minor(v: str) -> tuple[int, int]:
        parts = v.lstrip("v").split(".")
        return int(parts[0]), int(parts[1])

    try:
        c_major, c_minor = _major_minor(client_version)
        s_major, s_minor = _major_minor(server_version)
    except (ValueError, IndexError):
        logger.warning(
            "Could not parse Qdrant versions; skipping compatibility check",
            extra={"client": client_version, "server": server_version},
        )
        return

    if c_major != s_major or abs(c_minor - s_minor) > 1:
        raise RuntimeError(
            f"Qdrant client {client_version} is incompatible with server "
            f"{server_version}. Major versions must match and minor versions "
            f"must differ by at most 1. Writes will fail silently over gRPC — "
            f"documents appear to ingest and the index stays empty. Align the "
            f"image tag in docker-compose.yml with the qdrant-client pin in "
            f"requirements.txt."
        )

    logger.info(
        "Qdrant client/server versions compatible",
        extra={"client": client_version, "server": server_version},
    )


async def close_qdrant_client() -> None:
    """
    Call on application shutdown to cleanly close connections.

    Should be invoked in FastAPI's lifespan shutdown handler
    or equivalent application teardown hook.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        logger.info("Closing AsyncQdrantClient connection")
        await _qdrant_client.close()
        _qdrant_client = None
        logger.info("AsyncQdrantClient closed successfully")
