# M&A Due Diligence Intelligence Engine — Decisions Log

All architecture decisions that deviated from or resolved ambiguities in p4.md are recorded here.

---

## Decision 1: Qdrant Quantization Config
- **Date**: Session 1
- **Context**: p4.md specifies INT8 quantization but doesn't specify always_ram setting
- **Resolution**: Set `always_ram=True` on ScalarQuantization for fast retrieval. Trade-off: higher memory usage, lower latency.
- **Impact**: collection_manager.py

## Decision 2: BM25 via FastEmbed
- **Date**: Session 2
- **Context**: p4.md says "FastEmbed BM25" but doesn't specify exact class/method
- **Resolution**: Used `fastembed.SparseTextEmbedding(model_name="Qdrant/bm25")` which produces Qdrant-native SparseVector objects directly
- **Impact**: hybrid_search.py

## Decision 3: asyncio.Lock() creation timing
- **Date**: Session 2
- **Context**: p4.md warns "asyncio.Lock() requires running event loop" but doesn't specify where to create rate limiter locks
- **Resolution**: Rate limiters in BudgetTracker are created lazily via `_get_rate_limiter()` on first use, not at class definition time. This prevents the "no running event loop" error.
- **Impact**: budget_tracker.py, rate_limiter.py

## Decision 4: Agent 5 Heuristic vs LLM Split
- **Date**: Session 2
- **Context**: p4.md says "Primary: heuristics (~60% of queries), Fallback: LLM on ambiguity"
- **Resolution**: Implemented three heuristic paths: (1) clearly good (mean_score >= 0.7), (2) clearly bad (mean_score < 0.2), (3) ambiguous → LLM. Thresholds chosen empirically based on typical reranker score distributions.
- **Impact**: quality_assessor.py

## Decision 5: TOCTOU Budget Race Condition
- **Date**: Session 2
- **Context**: p4.md's budget tracker used separate _budget_available() + _increment() calls
- **Resolution**: Replaced with atomic _try_consume() using conditional UPDATE. Single SQL statement: `UPDATE ... SET used_today = used_today + 1 WHERE used_today < limit`. Prevents two concurrent requests from both passing the check and overshooting.
- **Impact**: budget_tracker.py

## Decision 6: gemini model string format
- **Date**: Session 2
- **Context**: p4.md references "gemini-3.5-flash" and "gemini-3.1-flash-lite" but LiteLLM model strings need prefix
- **Resolution**: Used "gemini/gemini-3.5-flash" and "gemini/gemini-3.1-flash-lite" format for LiteLLM routing. Marked with ⚠ VERIFY in code for runtime validation.
- **Impact**: budget_tracker.py

## Decision 7: Rewriter JSON key → State key mapping
- **Date**: Session 2
- **Context**: p4.md's Agent 6 output uses `updated_retrieval_config` and `updated_metadata_filters` but AgentState uses `retrieval_config` and `extracted_filters`
- **Resolution**: query_rewriter_node explicitly maps keys when returning partial state. Documented in prompt template docstring.
- **Impact**: query_rewriter.py

## Decision 8: Docker Compose Ollama connectivity
- **Date**: Session 2
- **Context**: Agents 4 and 8 use local Ollama which runs on the host, not in Docker
- **Resolution**: Set OLLAMA_API_BASE to `http://host.docker.internal:11434` in Docker Compose. This resolves to the host machine on both Docker Desktop (Mac/Windows) and newer Docker Engine (Linux with --add-host).
- **Impact**: docker-compose.yml

## Decision 9: Missing dependencies beyond p4.md requirements.txt
- **Date**: Session 3
- **Context**: p4.md's requirements.txt lists `asyncpg` and `psycopg2-binary` for Postgres but does NOT list `langgraph-checkpoint-postgres` or `psycopg[binary]`. At runtime, `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver` (used in orchestrator.py per p4.md's own code) fails with `ModuleNotFoundError: No module named 'langgraph.checkpoint.postgres'`. The `langgraph` pip package does NOT bundle the postgres checkpoint adapter — it's a separate package. Furthermore, `langgraph-checkpoint-postgres` depends on `psycopg` (v3, async-native), not `psycopg2-binary` (v2, sync-only). The binary wheel (`psycopg[binary]`) is needed on systems without `libpq` development headers.
- **Resolution**: Added `langgraph-checkpoint-postgres>=2.0.0,<3.0.0` and `psycopg[binary]>=3.1.0,<4.0.0` to requirements.txt. Pinned `<3.0.0` to avoid breaking changes with `langgraph>=0.2.0,<0.3.0` (version 3.x of checkpoint-postgres requires langgraph 0.3+). Validated: 76/76 tests pass after addition.
- **Impact**: requirements.txt

## Decision 10: Docker stub directory cleanup
- **Date**: Session 4
- **Context**: Phase 0 scaffold created `docker/Dockerfile`, `docker/docker-compose.yml`, and `docker/qdrant_config.yaml` as stubs. Phase 6 then placed the real implementations at project root (`Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml`) and the Qdrant config at `config/qdrant_config.yaml`. The `docker/` stubs were never updated and contained only placeholder comments.
- **Resolution**: Deleted `docker/` directory entirely. Fixed `README.md` reference from `docker compose -f docker/docker-compose.yml up -d` to `docker compose up -d`.
- **Impact**: docker/, README.md

## Decision 11: Streamlit import path and API base URL
- **Date**: Session 5
- **Context**: `streamlit run app/streamlit_app.py` crashed on its first import with `ModuleNotFoundError: No module named 'app'`. Streamlit's `bootstrap._fix_sys_path()` inserts only the *script's* directory (`app/`) into `sys.path`, never the project root, so the absolute `app.components.*` imports could not resolve — locally or in Docker. Separately, `API_URL` was hardcoded to `http://localhost:8000/api/v1` while `docker-compose.yml` injects `API_URL=http://api:8000`; inside the container `localhost` is the Streamlit container itself. Either bug alone made the Dockerized UI non-functional.
- **Resolution**: Prepend the resolved project root to `sys.path` in `streamlit_app.py` before the component imports (works identically bare-metal and in Docker, no env var required), and read the API base from `os.getenv("API_URL", "http://localhost:8000")`, appending `/api/v1`. Pinned Streamlit/requests in `Dockerfile.streamlit`, added a `/_stcore/health` healthcheck to the compose service, and copied `.streamlit/` into the image so the theme applies.
- **Impact**: app/streamlit_app.py, Dockerfile.streamlit, docker-compose.yml, .streamlit/config.toml

## Decision 12: include_pii lives on the state root, not in extracted_filters
- **Date**: Session 5
- **Context**: `QueryRequest.include_pii` was accepted by the API but never reached retrieval, so the flag was inert (failing safe, but over-promising in the contract). The obvious fix — put it in `extracted_filters` — is wrong: Agents 1 and 6 deliberately `pop("include_pii")` from anything an LLM produced, precisely so a model cannot escalate its own data access.
- **Resolution**: Added `include_pii` as a top-level `AgentState` field, set once by `run_query()` from the authenticated request. `retrieval_executor_node` merges it into the filter dict passed to `hybrid_search()` and forwards it to `expand_context()`. The LLM-stripping rule is untouched, and the query route records every authorized use on its `AUDIT_LOG` line.
- **Impact**: state_definitions.py, orchestrator.py, retrieval_executor.py, api/routes/query.py

## Decision 13: Document registry + version retirement
- **Date**: Session 5
- **Context**: `RiskSignalExtractor` and the risk/version UI components were fully implemented but unreachable — nothing produced their data. `document_count` on a deal never left 0. Worse, uploading a replacement document left *both* versions marked `is_current_version=1`, so retrieval's default version filter returned superseded terms alongside the ones that replaced them.
- **Resolution**: Added an in-process `_documents` registry in `api/routes/deals.py` (same pattern and same single-worker constraint as `_deals`), populated by the ingestion route, exposed via `GET /deals/{id}/documents` and `GET /deals/{id}/risk-signals`. Wired `RiskSignalExtractor` into ingestion — regex only, no LLM, no added latency — writing `risk_signals` per chunk and aggregating to document level. Ingesting with `supersedes_doc_id` now issues a Qdrant `set_payload` flipping the prior document's chunks to `is_current_version=0` and stamping `superseded_by`. Severity ranking lives in the API layer, not the extractor: the extractor reports what it matched, the dashboard ranks it.
- **Trade-off**: The registry is process-local and lost on restart, and it is now the only record of version chains and risk signals. Moving it to Postgres is the first step of the scale-out path.
- **Impact**: api/routes/deals.py, api/routes/ingest.py, api/models/response_models.py

## Decision 14: AsyncPostgresSaver lifetime
- **Date**: Session 5
- **Context**: `get_compiled_graph()` called `AsyncPostgresSaver.from_conn_string(url).setup()`. That factory returns an `_AsyncGeneratorContextManager`, which has no `.setup()` — so the call raised `AttributeError` on **every** startup and was swallowed by the `except` that falls back to `MemorySaver`. Postgres checkpointing had therefore never worked in any environment, and the warning it logged read like "Postgres is down" even when it was up. Durable checkpoints and crash recovery were silently absent.
- **Resolution**: Enter the context manager into a module-level `AsyncExitStack` that lives as long as the compiled graph, and unwind it from the FastAPI lifespan via a new `close_checkpointer()`. The `MemorySaver` fallback is retained for genuine connection failures.
- **Note**: Verified against the library contract (`from_conn_string` returns an async context manager with no `setup` attribute). The connected path could not be exercised locally — no Postgres instance was available — so first Docker run should be checked for `"compiled with PostgresSaver"` rather than the MemorySaver warning.
- **Impact**: src/workflow/orchestrator.py, api/main.py

## Decision 15: Escape currency before Streamlit renders LLM prose
- **Date**: Session 5
- **Context**: Streamlit's markdown parses `$…$` as inline LaTeX. An answer reading "revenue was $184.2M, up from $151.7M" had the span between the two dollar signs rendered as math: both `$` characters and the enclosing bold markers were silently swallowed, so figures disappeared from the rendered answer. Verified in the DOM (`class="language-math math-inline"`).
- **Resolution**: `app/styles.py::escape_currency()` normalizes then escapes `$` before any LLM-produced string reaches `st.markdown`. Applied to the answer body, hallucination flags, and risk descriptions.
- **Rationale**: For an engine whose stated premise is that a wrong number is a hard transaction failure, a number vanishing at the render layer is a correctness defect, not a styling one.
- **Impact**: app/styles.py, app/components/answer_display.py, app/components/risk_dashboard.py

## Decision 16: Citation precision — investigated, deliberately not changed
- **Date**: Session 5
- **Context**: With citations now carrying real metadata (Decision 12's sibling change), the UI displays section headings — and a live run exposed that they are wrong. An answer citing only "SECTION 8. INDEMNIFICATION" three times produced citations labelled *SECTION 7. LITIGATION*, *SECTION 5. CHANGE OF CONTROL*, and a mid-sentence fragment. Root cause: `answer_synthesizer` matches `source_file in answer`, which is document-level, so every retrieved chunk from that document is returned.
- **Attempted**: Narrowing to chunks whose `section_heading` also appears in the answer, falling back to file-level when none match. **Reverted** — it changed nothing in practice. `section_heading` is the heading of the *chunk*, and a chunk routinely spans several sections (a 998-byte test agreement produced 3 chunks covering 4 sections), so the answer's section heading is simply not present in the chunk metadata to match against.
- **Resolution**: Left the existing behaviour in place and documented the imprecision in-code, rather than shipping speculative logic with no demonstrated benefit. A real fix is one of: (a) per-chunk heading granularity from the chunker, or (b) parsing the inline `[file | p.N | SECTION]` citation markers the synthesis prompt already instructs the model to emit — (b) is cheaper and self-validating, but couples the parser to the prompt contract and needs its own tests.
- **Impact**: src/agents/answer_synthesizer.py (comment only)

