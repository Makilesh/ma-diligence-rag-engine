# Deployment

A permanently free, end-to-end deployment: Next.js frontend on Vercel, FastAPI
backend on a Hugging Face Space, Qdrant Cloud for vectors, Neon for Postgres,
and the existing Gemini free-tier key rotation for inference.

Nothing here is a trial or a credit that expires.

| Layer | Host | Free tier | Expires? |
|---|---|---|---|
| Frontend | Vercel Hobby | Unlimited static + edge | No — but non-commercial use only |
| Backend | Hugging Face Spaces (Docker, CPU basic) | 2 vCPU · 16GB RAM · 50GB disk | No — sleeps after 48h idle |
| Vectors | Qdrant Cloud | 1GB cluster, no card required | No |
| Postgres | Neon | 0.5GB, autosuspends when idle | No |
| LLM | Gemini API | Per-key daily quota, multiplied by key count | No |

---

## Why this shape, and not a smaller one

The backend is the constraint, and it is not close. `bge-m3` alone is 2.27GB
resident. The 512MB free tiers at Render, Fly and Koyeb are not near-misses —
they are an order of magnitude short. Hugging Face Spaces is the only free host
that fits the workload, and it fits it comfortably.

**The reranker had to change.** Measured on 2 vCPU, which is what the free tier
provides:

| Reranker | `max_length` | 40 passages | ×4 passes (a decomposed query) |
|---|---|---|---|
| `bge-reranker-v2-m3` (568M) | 1024 | 111.2s | **444.9s** |
| `bge-reranker-v2-m3` (568M) | 512 | 67.8s | 271.2s |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) | 512 | **3.2s** | **12.9s** |

Seven and a half minutes per question is not a demo. Quantization does not close
that gap — int8 buys 2–4× where ~30× is needed — so the deployed profile uses a
smaller cross-encoder, set through `RERANKER_MODEL`.

Two consequences worth stating plainly:

- **The deployed reranker is English-only.** The data room is English, so this
  costs nothing here, but it is a real capability difference from the local
  configuration.
- **The retrieval numbers in `RESULTS.md` were measured with the default
  reranker.** They do not describe the deployed profile until the eval is re-run
  against it (last step below).

**The embedding model does not change.** It costs 0.12s per query — it was never
the problem — and swapping it would invalidate every vector already indexed.
That is why no re-indexing is needed when moving between profiles.

---

## 1 · Qdrant Cloud

1. Create a free cluster at <https://cloud.qdrant.io>. No card required.
2. From **Connect**, copy the cluster URL and create an API key.

The API key does double duty: `src/vector_db/qdrant_client.py` treats its
presence as the signal that this is a managed cluster and switches the transport
from gRPC to REST. Managed clusters terminate TLS on the REST port and do not
expose plain gRPC on 6334, so a client that prefers gRPC there constructs
successfully and then fails on first use.

## 2 · Seed the index

The corpus is ingested once, from your machine, directly into Qdrant Cloud. The
Space never needs the documents — it only queries the vectors.

Point your local `.env` at the cloud cluster:

```bash
QDRANT_URL=https://<cluster>.cloud.qdrant.io:6333
QDRANT_API_KEY=<your key>
```

Then start the API locally and push the sample data room through the existing
ingest endpoint:

```bash
python run_api.py
```

```bash
for f in data/sample_deal/*.txt; do curl -sS -F "deal_id=aurora_vertex_2024" -F "file=@$f" http://localhost:8000/api/v1/ingest; echo; done
```

Writing to a non-sandbox deal is an admin operation. A local API started with
`ENVIRONMENT=development` and no `ADMIN_API_KEY` allows it; if you have set a
key, add `-H "X-Admin-Key: $ADMIN_API_KEY"` to the curl above. Re-running the
loop is safe — document and point IDs are derived from content, so identical
files replace themselves instead of duplicating.

Confirm it landed:

```bash
curl -s http://localhost:8000/api/v1/deals
```

Nine documents, roughly 150 chunks — well inside the 1GB tier.

## 3 · Neon Postgres

Create a free project at <https://neon.tech> and copy the pooled connection
string. This backs the daily budget tracker and LangGraph checkpoints.

It is optional. Without it both degrade to in-memory: the pipeline stays fully
functional, but daily quota accounting resets on every restart — which on a
sleeping Space is often.

## 4 · Hugging Face Space

1. Create a new Space → **Docker** → **Blank**, hardware **CPU basic (free)**.
2. Add this to the Space's `README.md` so it builds the right file on port 7860:

   ```yaml
   ---
   title: M&A Due Diligence Intelligence Engine
   sdk: docker
   app_port: 7860
   dockerfile_path: Dockerfile.hf
   ---
   ```

3. Push this repository to the Space's git remote.
4. Under **Settings → Variables and secrets**, add everything from
   `.env.deploy.example`. Credentials go in **Secrets**; the rest in
   **Variables**.
5. Set the public-demo guardrails (`api/security.py`): a long random
   `ADMIN_API_KEY` as a **Secret**, and `TRUST_PROXY_HEADERS=1` as a Variable so
   per-IP rate limits see the visitor's address rather than the Space's proxy.
   Without an admin key the Space runs in public mode: visitors can query and
   use their own sandbox, but cannot write to or delete the demo deal.

The first build takes a while — it bakes the model weights into the image on
purpose. The free tier's disk is ephemeral, so a runtime download would be paid
again after every sleep-wake cycle, in front of whoever opened the link.

## 5 · Vercel

1. Import the repository at <https://vercel.com/new>.
2. Set **Root Directory** to `web`. Everything else is auto-detected.
3. Set `NEXT_PUBLIC_API_URL` to your Space URL
   (`https://<user>-<space>.hf.space`, no trailing slash).

`NEXT_PUBLIC_*` is inlined at build time, so changing it requires a redeploy —
not a restart.

## 6 · Close the CORS loop

Set `CORS_ORIGINS` on the Space to the Vercel URL and restart it.

Get this wrong and the failure is silent on the server: the browser blocks every
request, the UI shows an empty deal list, and the Space's logs show nothing at
all, because the requests never arrive.

## 7 · Keep the Space awake

`.github/workflows/keepalive.yml` pings `/health` every 6 hours, well inside the
48-hour sleep window.

Set a repository **variable** (not a secret — a Space URL is public) named
`SPACE_URL` under *Settings → Secrets and variables → Actions → Variables*, then
run the workflow once manually to confirm it resolves.

**This needs occasional attention.** GitHub disables scheduled workflows in a
repository with no activity for 60 days. When that happens the ping stops
silently and the Space starts sleeping again. Any commit resets the clock.

## 8 · Re-run the eval against the deployed profile

The last step, and the one that keeps the project's headline claims honest.

**Pin the synthesis model, or the comparison is worthless.** Recall on this set
is dominated by which model answered, not by retrieval. Measured inside a single
run — one index, identical retrieval — mean recall was:

| Synthesis model | n | mean recall |
|---|---|---|
| `gemini-3.6-flash` | 1 | 100.0% |
| `gemini-3-flash-preview` | 10 | 93.3% |
| `gemini-3.5-flash` | 20 | 65.8% |
| `gemini-2.5-flash` | 4 | 50.0% |

A 50-point spread from model availability alone — and it is not quota that
causes it. With 5 keys each rung has ~95 requests/day, and after two full eval
runs `gemini-3.7-flash` still had 86 left. What actually moves the mix is Google
returning 503 "this model is currently experiencing high demand", which puts the
model into an escalating process-wide cooldown (90s doubling to 720s). One rung
sat out 12 minutes of a 25-minute run that way, and synthesis fell to whatever
was still answering.

So an unpinned local-vs-deployed comparison measures which models Google happened
to be serving that hour, at least as much as it measures the reranker.

`SYNTHESIS_MODEL_PIN` restricts synthesis to one model and never substitutes.
It distinguishes the two ways a pin can be unavailable: a 503 cooldown is
transient, so it waits (up to `SYNTHESIS_PIN_MAX_WAIT_SECONDS`, default 1800) and
retries; exhausted daily quota is terminal, so it raises immediately.

**Which model to pin.** `gemini-3.7-flash` is the top of the ladder and had the
most headroom in practice, so it is the better pin for new measurements. Note
that the historical figures in `RESULTS.md` were produced on `gemini-3.6-flash`
— comparing against those specifically requires pinning to that instead.

**Set these on the API server, not on the eval command.** The harness is a pure
HTTP client — model selection, retrieval and reranking all happen inside the
FastAPI process, so an env var exported for the harness reaches nothing that
acts on it. This is easy to get wrong and fails silently: the run completes and
reports an unpinned mix.

Use the *same* pin for both runs, on a day with quota to spare:

```bash
SYNTHESIS_MODEL_PIN=gemini/gemini-3.7-flash python run_api.py
```

```bash
python tests/run_end_to_end_validation.py
```

Then restart the API with the deployed retrieval profile and repeat:

```bash
SYNTHESIS_MODEL_PIN=gemini/gemini-3.7-flash RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 RERANKER_MAX_LENGTH=512 python run_api.py
```

The report records both the retrieval profile and the pin under **Run
conditions**, so the two runs are self-labelling rather than two undated tables.

`RESULTS.md` and the README currently report one configuration. Once the live
demo runs a different reranker, they describe two, and both should be stated —
the local profile the accuracy was measured on, and the deployed profile a
visitor actually exercises.

The reranker thresholds in `src/agents/retrieval_strategy.py` (0.25–0.4) were
tuned against `bge-reranker-v2-m3`'s sigmoid output. A different cross-encoder's
scores are on a comparable but not identical scale, so if the eval shows more
refusals than expected, those thresholds are the first place to look.

---

## What the deployment does not have

**The local Ollama fallback is gone.** The model ladder's last rung is a local
Qwen2.5-14B served by Ollama, which exists to answer when every cloud key is
spent. A free Space has no Ollama server, and 14B weights would not fit
alongside the retrieval models even if it did.

The practical effect: on the deployed instance, exhausting the Gemini daily
quota is terminal for that day rather than a quiet downgrade. Since keys
multiply capacity and rotation drains one key per model before stepping down a
rung, the way to push that ceiling out is more keys in `GEMINI_API_KEYS`, not a
different host.

## Not yet verified

- **The `Dockerfile.hf` build has not been run.** Docker was unavailable on the
  machine this was written on, so the file is written against the platform's
  documented requirements (port 7860, uid 1000, `HF_HOME` under the user's home)
  but has not been built. Expect to iterate on the first Space build.
- **Reranker threshold recalibration** — see step 8.
