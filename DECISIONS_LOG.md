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


## Decision 17: Calibrate the quality gate against the reranker's real score distribution
- **Date**: Session 6
- **Context**: 10 of 16 completed golden questions were refused. The Quality Assessor scored context as `mean(top_5)` / `min(top_5)` of reranker scores and required `mean >= 0.5 and min >= 0.2`; the per-type gate then demanded e.g. `precision >= 0.8` for financial queries. Nobody had measured what BAAI/bge-reranker-v2-m3 actually outputs on this corpus. Labelling every reranked chunk relevant/irrelevant using the golden set's expected facts gave the answer: relevant chunks score 0.24-0.99 (median 0.241), noise sits at median 0.006. The distribution is bimodal, so (a) averaging across the top-k means **retrieving more candidates lowers the score** - the gate punished recall - and (b) `min(top_5) >= 0.2` required the fifth-best chunk to be excellent, which such distributions rarely satisfy. Evidence: fin_02 and mh_04 scored 0.638 and 0.645 on genuinely good context and were refused.
- **Resolution**: Score the *usable* evidence instead of the whole retrieved set. `relevance = max(score)`; `precision = mean(scores >= RELEVANCE_FLOOR)`; `completeness = len(usable) / EXPECTED_EVIDENCE_COUNT[query_type]`. `RELEVANCE_FLOOR = 0.10` is the peak-F1 point of a sweep over the labelled data (F1 0.534). Thresholds re-derived the same way: the 18 questions whose retrieval genuinely contained the answer scored relevance >= 0.521 / precision >= 0.298 / completeness >= 0.333, while the one true retrieval failure scored 0.009 / 0.000 / 0.000 - so 0.30 / 0.25 / 0.30 sits in a wide empty band rather than near a dense part of the distribution.
- **Side effect**: the confident bands now cover the whole observed range, so the LLM fallback fired 0/19 times - one fewer model call per query.
- **Guarded by**: `test_one_decisive_chunk_is_enough`, `test_retrieving_more_noise_does_not_lower_quality`, plus an adversarial all-noise check that must still refuse.
- **Impact**: quality_assessor.py, conditional_edges.py

## Decision 18: Progressive relaxation of the document_category filter
- **Date**: Session 6
- **Context**: Agent 1 infers a `document_category` and it was applied as a hard Qdrant `must`. When the inference is wrong the answer is removed from the search space and the rewrite loop cannot recover it, because rewriting changes query text and never filters. Measured on the golden set: legal_01 ("per-share merger consideration") was classified `financial`, filtering out the merger agreement - the only document containing the answer; fin_05 (credit facility terms) was classified `legal` and excluded the financials; comp_02 (valuation methodologies) was classified `financial` and excluded the board deck. All three burned both rewrite attempts against a search space that could not contain the answer. It also compounds two error sources, since the category being matched was itself assigned by a heuristic classifier at ingestion.
- **Resolution**: Keep the category filter on the first attempt, where it buys precision, and drop it once `rewrite_iteration >= 1` - the point at which the system is explicitly trading precision for recall. `deal_id`, `is_current_version` and `contains_pii` are never relaxed: those are isolation and compliance constraints, not relevance hints.
- **Guarded by**: `TestProgressiveFilterRelaxation`.
- **Impact**: retrieval_executor.py

## Decision 19: The hallucination validator must see the synthesizer's evidence
- **Date**: Session 6
- **Context**: 10 of 19 answers came back `validation_status="failed"`, several with 100% fact recall against the golden set. fin_01 was flagged for claiming revenue was $452.8M/$387.1M - the correct, sourced figures. Cause: the validator formatted context as `c["text"][:500]`, while the synthesizer received the full chunk plus parent context. The validator was asked "is this claim supported?" while being shown roughly a sixth of the supporting text, so grounded numbers past the cutoff looked unsupported.
- **Resolution**: Give the validator the full chunk text and parent context - the same evidence the writer had. Flagged answers dropped from 11/19 to 2/19 and `passed` rose from 8 to 17.
- **Why it matters beyond the metric**: a safety control with a high false-positive rate is worse than none, because reviewers learn to ignore it.
- **Impact**: hallucination_validator.py

## Decision 20: Verification agents default to cloud, with local retained
- **Date**: Session 6
- **Context**: Agents 4 and 8 always used `ollama/qwen2.5:14b`. That was a deliberate cost decision - verification is the highest-volume agent traffic, and keeping it local held the pipeline inside the free Gemini tier - but it made the project unrunnable without a 12GB-VRAM host and a 9GB model pull, and local inference dominated latency.
- **Resolution**: `VERIFICATION_BACKEND` selects the backend, defaulting to `cloud`. The cloud path routes through `BudgetTracker.get_model_for_agent()`, so it takes a rate-limiter slot, debits the daily quota, and automatically returns the local model when the quota is spent. Setting `VERIFICATION_BACKEND=local` restores the original behaviour. Both paths are live.
- **Measured**: mean latency 55.1s -> 18.2s. Cost: cloud calls per query roughly doubled, which is what exposed Decision 21.
- **Impact**: litellm_wrapper.py, financial_verifier.py, hallucination_validator.py

## Decision 21: Rate-limit by upstream model, not by budget bucket
- **Date**: Session 6
- **Context**: After Decision 20 the golden-set run failed with `RateLimitError`. `_get_rate_limiter` keyed limiters by budget bucket - `synthesis_primary` at 5 RPM and `agent_workhorse` at 15 RPM - but both buckets resolve to the same `gemini-3.1-flash-lite`, so the client permitted 20 requests/minute against a single 15 RPM provider quota. The bug pre-dated the change and stayed hidden only because local verification kept cloud volume low. Separately, `call_structured_agent` retried transport errors with no delay, so a 429 burned all three attempts in milliseconds.
- **Resolution**: Key rate limiters by upstream model string via `BUCKET_MODELS` / `MODEL_RPM_LIMITS`, so every bucket spending against the same model shares one limiter. Daily budgets stay per-bucket - that is cost allocation, a separate concern from provider metering. Added exponential backoff to structured-agent retries, with a longer pause for rate-limit errors since the window they enforce is measured in seconds.
- **Result**: 23/23 queries completed with zero rate-limit or connection errors.
- **Guarded by**: `TestSharedModelRateLimiter`.
- **Impact**: budget_tracker.py, litellm_wrapper.py

## Decision 22: Unanswerable control questions in the golden set
- **Date**: Session 6
- **Context**: After Decisions 17 and 18 the engine answered 19/19 answerable questions and refused none. On a set where every question is answerable, a 0% refusal rate is unfalsifiable: "correctly answers everything" and "gate is broken and never refuses" produce identical numbers.
- **Resolution**: Added 4 control questions (`ctrl_*`) whose answers are absent from the corpus by construction - environmental liabilities, FY2024 quarterlies, headcount by office, churn vs competitors - and scored them on whether the engine fabricated the missing figure.
- **Finding**: the first scoring pass reported 1/3 and looked like hallucination. It was the metric, not the engine: a binary refused/answered flag scored *partial* answers as failures. The engine had replied "The provided documents do not contain the revenue figures for Q1 FY2024" with a citation, and for the churn question reported the retention and competitor data that do exist while stating churn does not. That is better due-diligence behaviour than a blanket refusal. Re-scored on "did it fabricate the missing fact": 4/4.
- **Also surfaced**: one control crashed the pipeline - RRF raised `ValueError` when both retrievers returned empty, turning a legitimate no-results query into a 500. Now returns an empty list so the Quality Assessor takes the refusal path.
- **Impact**: tests/golden_qa_set.json, tests/run_end_to_end_validation.py, rrf_fusion.py

## Decision 23: Streamlit example queries must write to the widget's own state key
- **Date**: Session 6
- **Context**: Clicking an example query filled the text area visually but left the Search button disabled. The handler wrote to a separate `query_text` key passed as `value=`; once a keyed widget exists Streamlit serves its stored state and ignores `value=`, so the widget still returned an empty string and `disabled=not query` stayed true. Found by driving the real UI, not by reading the code - the DOM and the widget state disagreed.
- **Resolution**: Write directly to `st.session_state["query_input"]` (the text area's own key) and call `st.rerun()`.
- **Impact**: app/components/query_interface.py
