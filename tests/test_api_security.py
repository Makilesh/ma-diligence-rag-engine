"""
Public-demo guardrails on the FastAPI app: rate limits, the pipeline gate, admin
gating of destructive operations, sandbox privacy and durability, PII policy,
error hygiene, input validation and the health/readiness probes.

The app is exercised through TestClient *without* entering its context manager,
so the lifespan never runs: no Qdrant connection, no Postgres, no graph
compilation and no model loading. Everything the routes reach for at request
time is stubbed instead — the Qdrant client singleton, the compiled graph and
the orchestrator's run/stream functions — which leaves production wiring
untouched.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import api.main as api_main
import api.routes.deals as deals
import api.routes.query as query_routes
import api.security as security
import src.vector_db.qdrant_client as qdrant_module
from src.vector_db.constants import COLLECTION_NAME

DEMO_DEAL = "aurora_vertex_2024"
ADMIN_KEY = "test-admin-key"


# ==============================================================================
# Stubs
# ==============================================================================


class FakeQdrant:
    """Just enough of AsyncQdrantClient for the deal routes and /ready."""

    def __init__(self, deals_to_files: dict[str, list[str]]):
        self.deals = {k: list(v) for k, v in deals_to_files.items()}
        self.deleted: list[str] = []
        self.reachable = True
        self.fail_delete = False

    async def facet(self, collection_name, key, facet_filter=None, limit=1000):
        if key == "deal_id":
            values = list(self.deals)
        else:
            deal_id = facet_filter.must[0].match.value
            values = self.deals.get(deal_id, [])
        return SimpleNamespace(hits=[SimpleNamespace(value=v, count=1) for v in values])

    async def delete(self, collection_name, points_selector):
        if self.fail_delete:
            raise RuntimeError("delete failed at http://internal-qdrant:6333 token=abc")
        deal_id = points_selector.filter.must[0].match.value
        if collection_name == COLLECTION_NAME:
            self.deleted.append(deal_id)
            self.deals.pop(deal_id, None)

    async def scroll(self, **kwargs):
        return [], None

    async def get_collections(self):
        if not self.reachable:
            raise ConnectionError("qdrant down")
        return SimpleNamespace(collections=[])


def _sandbox_id(created: datetime) -> str:
    return deals.new_sandbox_id(created)


@pytest.fixture
def fake_qdrant(monkeypatch):
    fake = FakeQdrant({DEMO_DEAL: ["merger_agreement.pdf", "financials.pdf"]})
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: fake)
    return fake


@pytest.fixture
def pipeline_calls(monkeypatch):
    """Stubs the orchestrator and records the kwargs each run received."""
    calls: list[dict] = []

    async def fake_run_query(**kwargs):
        calls.append(kwargs)
        return {"generated_answer": "Answer [1].", "query_type": "summary", "status": "complete"}

    async def fake_stream_query(**kwargs):
        calls.append(kwargs)
        yield "start", {"session_id": kwargs["session_id"], "planned_stages": []}
        yield "result", {"generated_answer": "Streamed.", "status": "complete"}

    monkeypatch.setattr(query_routes, "run_query", fake_run_query)
    monkeypatch.setattr(query_routes, "stream_query", fake_stream_query)
    return calls


@pytest.fixture(autouse=True)
def public_environment(monkeypatch):
    """
    Every test starts as an anonymous public caller in production mode.

    Set explicitly rather than inherited: importing the app loads `.env` as a
    side effect of importing litellm, and a developer's `.env` normally carries
    ENVIRONMENT=development — which would make every caller admin.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    # Generous by default so only the tests about limits ever hit one.
    monkeypatch.setenv("RATE_LIMIT_QUERY_PER_MINUTE", "1000")
    monkeypatch.setenv("RATE_LIMIT_QUERY_PER_DAY", "1000")
    monkeypatch.setenv("DAILY_QUERY_CAP", "1000")
    monkeypatch.setenv("MAX_CONCURRENT_PIPELINES", "2")
    monkeypatch.setattr(api_main, "_app_graph", object())
    security.reset_limits()
    deals._deals.clear()
    deals._documents.clear()
    yield
    security.reset_limits()
    deals._deals.clear()
    deals._documents.clear()


@pytest.fixture
def client():
    # No `with`: entering the context would run the real lifespan.
    return TestClient(api_main.app)


def _ask(client, deal_id=DEMO_DEAL, headers=None, **extra):
    body = {"query": "What is the revenue?", "deal_id": deal_id, **extra}
    return client.post("/api/v1/query", json=body, headers=headers or {})


# ==============================================================================
# Rate limiting, daily cap, concurrency
# ==============================================================================


def test_query_rate_limit_returns_429_after_limit(client, pipeline_calls, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_QUERY_PER_MINUTE", "2")

    assert _ask(client).status_code == 200
    assert _ask(client).status_code == 200
    blocked = _ask(client)

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert len(pipeline_calls) == 2, "a rate-limited request must not reach the pipeline"


def test_stream_endpoint_shares_the_query_bucket(client, pipeline_calls, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_QUERY_PER_MINUTE", "1")
    body = {"query": "q", "deal_id": DEMO_DEAL}

    assert client.post("/api/v1/query/stream", json=body).status_code == 200
    assert client.post("/api/v1/query/stream", json=body).status_code == 429
    assert _ask(client).status_code == 429


def test_per_day_limit(client, pipeline_calls, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_QUERY_PER_DAY", "1")
    assert _ask(client).status_code == 200
    assert _ask(client).status_code == 429


def test_forwarded_for_is_only_trusted_when_configured(client, pipeline_calls, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_QUERY_PER_MINUTE", "1")

    # Untrusted: a client-chosen X-Forwarded-For must not buy a fresh bucket.
    assert _ask(client, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert _ask(client, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429

    security.reset_limits()
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    # Trusted: the first hop identifies the client, so these are two buckets.
    assert _ask(client, headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}).status_code == 200
    assert _ask(client, headers={"X-Forwarded-For": "2.2.2.2, 10.0.0.1"}).status_code == 200
    assert _ask(client, headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}).status_code == 429


def test_admin_key_bypasses_rate_limits(client, pipeline_calls, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setenv("RATE_LIMIT_QUERY_PER_MINUTE", "1")
    admin = {security.ADMIN_HEADER: ADMIN_KEY}

    for _ in range(3):
        assert _ask(client, headers=admin).status_code == 200
    # A wrong key is just a public caller.
    assert _ask(client, headers={security.ADMIN_HEADER: "wrong"}).status_code == 200
    assert _ask(client, headers={security.ADMIN_HEADER: "wrong"}).status_code == 429


def test_global_daily_cap(client, pipeline_calls, monkeypatch):
    monkeypatch.setenv("DAILY_QUERY_CAP", "2")
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")

    # Different clients, so only the global cap can be what stops the third.
    assert _ask(client, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert _ask(client, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    capped = _ask(client, headers={"X-Forwarded-For": "3.3.3.3"})

    assert capped.status_code == 429
    assert "daily" in capped.json()["detail"].lower()


def test_concurrency_cap_fails_fast_with_503(client, pipeline_calls, monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_PIPELINES", "1")
    holder = security.ClientContext(ip="9.9.9.9", is_admin=False, request_id="held")
    slot = security.acquire_pipeline_slot(holder)

    busy = _ask(client)
    assert busy.status_code == 503
    assert busy.headers.get("Retry-After")
    assert not pipeline_calls

    slot.release()
    assert _ask(client).status_code == 200


def test_slots_are_released_after_blocking_and_streamed_queries(client, pipeline_calls):
    assert _ask(client).status_code == 200
    stream = client.post("/api/v1/query/stream", json={"query": "q", "deal_id": DEMO_DEAL})
    assert stream.status_code == 200
    assert "event: result" in stream.text
    assert security._gate.in_flight == 0


def test_slot_released_when_pipeline_raises(client, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(query_routes, "run_query", boom)
    assert _ask(client).status_code == 500
    assert security._gate.in_flight == 0


# ==============================================================================
# Admin gating of destructive operations
# ==============================================================================


def test_purge_of_non_sandbox_deal_requires_admin(client, fake_qdrant, monkeypatch):
    assert client.delete(f"/api/v1/deals/{DEMO_DEAL}").status_code == 403
    assert client.post(f"/api/v1/deals/{DEMO_DEAL}/purge").status_code == 403
    assert DEMO_DEAL in fake_qdrant.deals, "the demo corpus must survive an anonymous purge"

    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    assert (
        client.delete(
            f"/api/v1/deals/{DEMO_DEAL}", headers={security.ADMIN_HEADER: "wrong"}
        ).status_code
        == 403
    )
    ok = client.delete(f"/api/v1/deals/{DEMO_DEAL}", headers={security.ADMIN_HEADER: ADMIN_KEY})
    assert ok.status_code == 200
    assert fake_qdrant.deleted == [DEMO_DEAL]


def test_development_mode_without_key_is_admin(client, fake_qdrant, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert client.delete(f"/api/v1/deals/{DEMO_DEAL}").status_code == 200

    # Once a key is configured, development mode no longer implies admin.
    fake_qdrant.deals[DEMO_DEAL] = ["x.pdf"]
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    assert client.delete(f"/api/v1/deals/{DEMO_DEAL}").status_code == 403


def test_sandbox_deal_can_be_created_and_purged_publicly(client, fake_qdrant):
    created = client.post(
        "/api/v1/deals", json={"deal_name": "Sandbox", "is_sandbox": True}
    )
    assert created.status_code == 200
    deal = created.json()
    assert deal["is_sandbox"] and deals.is_sandbox_id(deal["deal_id"])
    assert deal["expires_at"]

    fake_qdrant.deals[deal["deal_id"]] = ["upload.pdf"]
    purged = client.post(f"/api/v1/deals/{deal['deal_id']}/purge")
    assert purged.status_code == 200
    assert purged.json()["status"] == "purged"
    assert deal["deal_id"] not in fake_qdrant.deals


def test_public_caller_cannot_create_a_permanent_deal(client):
    denied = client.post("/api/v1/deals", json={"deal_name": "Permanent"})
    assert denied.status_code == 403


def test_sandbox_ids_are_unguessable():
    now = datetime.now(timezone.utc)
    ids = {deals.new_sandbox_id(now) for _ in range(200)}
    assert len(ids) == 200
    random_part = next(iter(ids)).rsplit("-", 1)[1]
    assert len(random_part) == 32  # 128 bits from `secrets`


# ==============================================================================
# Sandbox privacy and durability
# ==============================================================================


def test_deal_listing_hides_sandboxes_from_public(client, fake_qdrant, monkeypatch):
    live_sandbox = _sandbox_id(datetime.now(timezone.utc))
    fake_qdrant.deals[live_sandbox] = ["someone_elses_upload.pdf"]
    in_memory = client.post(
        "/api/v1/deals", json={"deal_name": "Mine", "is_sandbox": True}
    ).json()["deal_id"]

    public_ids = {d["deal_id"] for d in client.get("/api/v1/deals").json()}
    assert public_ids == {DEMO_DEAL}

    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    admin_list = client.get("/api/v1/deals", headers={security.ADMIN_HEADER: ADMIN_KEY}).json()
    admin_ids = {d["deal_id"] for d in admin_list}
    assert {DEMO_DEAL, live_sandbox, in_memory} <= admin_ids
    discovered = next(d for d in admin_list if d["deal_id"] == live_sandbox)
    assert discovered["is_sandbox"] and discovered["expires_at"]


def test_sweeper_reclaims_orphaned_sandboxes_after_restart(fake_qdrant):
    now = datetime.now(timezone.utc)
    ttl = timedelta(seconds=deals.SANDBOX_TTL_SECONDS)
    expired = _sandbox_id(now - ttl - timedelta(minutes=1))
    live = _sandbox_id(now)
    forged_future = _sandbox_id(now + timedelta(days=30))
    for deal_id in (expired, live, forged_future):
        fake_qdrant.deals[deal_id] = ["upload.pdf"]

    # `_deals` is empty, exactly as it is after a restart.
    purged = asyncio.run(deals._sweep_expired_sandboxes())

    assert purged == 2
    assert set(fake_qdrant.deleted) == {expired, forged_future}
    assert live in fake_qdrant.deals and DEMO_DEAL in fake_qdrant.deals


def _raw_request(headers: dict[str, str] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            "client": ("5.5.5.5", 1234),
        }
    )


def test_require_sandbox_deal_helper(monkeypatch):
    from fastapi import HTTPException

    live = _sandbox_id(datetime.now(timezone.utc))
    stale = _sandbox_id(datetime.now(timezone.utc) - timedelta(days=2))

    asyncio.run(security.require_sandbox_deal(live, _raw_request()))

    for deal_id, status in ((DEMO_DEAL, 403), (stale, 410), ("../../etc", 422)):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(security.require_sandbox_deal(deal_id, _raw_request()))
        assert exc.value.status_code == status

    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    asyncio.run(
        security.require_sandbox_deal(DEMO_DEAL, _raw_request({security.ADMIN_HEADER: ADMIN_KEY}))
    )


# ==============================================================================
# PII policy
# ==============================================================================


def test_include_pii_forced_false_for_public_callers(client, pipeline_calls, monkeypatch):
    assert _ask(client, include_pii=True).status_code == 200
    client.post(
        "/api/v1/query/stream", json={"query": "q", "deal_id": DEMO_DEAL, "include_pii": True}
    )
    assert [c["include_pii"] for c in pipeline_calls] == [False, False]

    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    _ask(client, include_pii=True, headers={security.ADMIN_HEADER: ADMIN_KEY})
    assert pipeline_calls[-1]["include_pii"] is True


# ==============================================================================
# Error hygiene
# ==============================================================================

SECRET_ERROR = "connection refused at http://10.0.0.7:6333 api_key=sk-live-123"


def test_query_errors_do_not_leak_exception_text(client, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError(SECRET_ERROR)

    monkeypatch.setattr(query_routes, "run_query", boom)
    response = _ask(client)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "10.0.0.7" not in detail and "sk-live" not in detail
    request_id = response.headers[security.REQUEST_ID_HEADER]
    assert request_id and request_id in detail


def test_stream_error_event_does_not_leak_exception_text(client, monkeypatch):
    async def boom_stream(**kwargs):
        yield "start", {"session_id": "s", "planned_stages": []}
        raise RuntimeError(SECRET_ERROR)

    monkeypatch.setattr(query_routes, "stream_query", boom_stream)
    response = client.post("/api/v1/query/stream", json={"query": "q", "deal_id": DEMO_DEAL})

    assert "event: error" in response.text
    assert "10.0.0.7" not in response.text and "sk-live" not in response.text
    error_frame = response.text.split("event: error\ndata: ", 1)[1].split("\n", 1)[0]
    assert json.loads(error_frame)["request_id"] == response.headers[security.REQUEST_ID_HEADER]
    assert security._gate.in_flight == 0


def test_purge_errors_do_not_leak_exception_text(client, fake_qdrant):
    sandbox = _sandbox_id(datetime.now(timezone.utc))
    fake_qdrant.fail_delete = True
    response = client.delete(f"/api/v1/deals/{sandbox}")

    assert response.status_code == 500
    assert "internal-qdrant" not in response.text and "token" not in response.text


# ==============================================================================
# Input validation
# ==============================================================================


@pytest.mark.parametrize(
    "overrides",
    [
        {"deal_id": "../../etc/passwd"},
        {"deal_id": "a" * 65},
        {"deal_id": ""},
        {"deal_id": "deal id with spaces"},
        {"session_id": "x;DROP TABLE"},
        {"query": "q" * 2001},
        {"query": ""},
    ],
)
def test_query_validation_rejects_bad_input(client, pipeline_calls, overrides):
    body = {"query": "What is the revenue?", "deal_id": DEMO_DEAL, **overrides}
    assert client.post("/api/v1/query", json=body).status_code == 422
    assert not pipeline_calls


def test_path_deal_id_validated(client, fake_qdrant):
    assert client.get("/api/v1/deals/bad$id/documents").status_code == 422
    assert client.get("/api/v1/deals/" + "a" * 65).status_code == 422
    assert client.delete("/api/v1/deals/bad$id").status_code == 422


def test_deal_name_length_limited(client):
    too_long = {"deal_name": "x" * 201, "is_sandbox": True}
    assert client.post("/api/v1/deals", json=too_long).status_code == 422


# ==============================================================================
# GET /deals/{id} consistency
# ==============================================================================


def test_get_deal_resolves_indexed_deals_not_in_memory(client, fake_qdrant):
    listed = {d["deal_id"] for d in client.get("/api/v1/deals").json()}
    assert DEMO_DEAL in listed

    found = client.get(f"/api/v1/deals/{DEMO_DEAL}")
    assert found.status_code == 200
    assert found.json()["document_count"] == 2

    assert client.get("/api/v1/deals/does_not_exist").status_code == 404


# ==============================================================================
# Probes
# ==============================================================================


def test_health_is_static(client, monkeypatch):
    monkeypatch.setattr(api_main, "_app_graph", None)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "manda-rag"}


def test_ready_reports_graph_and_qdrant(client, fake_qdrant, monkeypatch):
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"graph_compiled": True, "qdrant_reachable": True}

    fake_qdrant.reachable = False
    assert client.get("/ready").status_code == 503

    fake_qdrant.reachable = True
    monkeypatch.setattr(api_main, "_app_graph", None)
    not_ready = client.get("/ready")
    assert not_ready.status_code == 503
    assert not_ready.json()["checks"]["graph_compiled"] is False
