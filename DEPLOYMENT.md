# Deployment

A free, end-to-end public deployment: the Next.js frontend on Vercel, and the
FastAPI backend, Qdrant and Postgres together on one Linux VM behind Caddy,
which provides HTTPS automatically.

```
Vercel (web/, free)  ──HTTPS──▶  https://<name>.duckdns.org
                                        │
                           Linux VM (≥ 2 cores / 8GB RAM)
                     Caddy :443 → api:7860 → qdrant, postgres
                          (docker-compose.prod.yml)
```

| Layer | Host | Cost |
|---|---|---|
| Frontend | Vercel Hobby | Free (non-commercial use) |
| Backend + Qdrant + Postgres | Oracle Cloud Always Free ARM (2 OCPU / 12GB), or any VM with ≥ 8GB RAM | Free on Oracle; card verification required at sign-up |
| Hostname + HTTPS | DuckDNS + Caddy (Let's Encrypt) | Free |
| LLM | Gemini API free tier | Per-key daily quota, multiplied by key count |

Oracle only reclaims an Always Free VM as idle when CPU, network *and* memory
all stay under 20% for a week; with the models loaded, memory alone sits well
above that.

---

## Why this shape, and not a smaller one

The backend is the constraint, and it is not close. `bge-m3` alone is 2.27GB
resident and the whole API holds about 5–6GB once the reranker and NLI model are
loaded. The 512MB free tiers at Render, Fly and Koyeb are not near-misses — they
are an order of magnitude short. What fits is a VM with at least 8GB of RAM:
Oracle Cloud's Always Free ARM instance (2 OCPU / 12GB) is the free option, and
co-locating Qdrant and Postgres on it removes two external accounts.

**The reranker had to change.** Measured on 2 vCPU, which is what a free VM
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

## 1 · A server with a public IP

Any Ubuntu 22.04/24.04 VM with a public IPv4 address, at least 2 cores, 8GB RAM
and 40GB of disk. On Oracle Cloud: Compute → Instances → Create instance →
Canonical Ubuntu 24.04, shape **VM.Standard.A1.Flex** at 2 OCPU / 12GB (it shows
"Always Free-eligible"), assign a public IPv4 address, and paste your SSH public
key. "Out of host capacity" is common — retry later or in another availability
domain.

A home connection does not work as the server. Mobile and most residential
broadband sit behind carrier-grade NAT and change address, so nothing on the
internet can reach a port on them.

## 2 · Open ports 80 and 443 — in both firewalls

- **Cloud firewall.** On Oracle: Networking → Virtual cloud networks → your VCN →
  Security Lists → Default → Add Ingress Rules: source `0.0.0.0/0`, TCP,
  destination port `80`; repeat for `443`.
- **Host firewall.** Oracle's Ubuntu image rejects everything but SSH in
  iptables, and skipping this is the usual reason a site is unreachable with no
  error anywhere:

  ```bash
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
  sudo netfilter-persistent save
  ```

Do not open 5432, 6333 or 6334. The production compose file publishes nothing
but Caddy's ports.

## 3 · A hostname

Create a subdomain at <https://www.duckdns.org> and set its IP to the **server's**
public IP. DuckDNS pre-fills the address you are browsing from, which is your
own connection, not the server — overwrite it. Check with:

```bash
nslookup <name>.duckdns.org   # must print the VM's public IP
```

Caddy cannot obtain a certificate until this resolves to the server.

## 4 · Docker and the code

```bash
ssh -i ~/.ssh/<key> ubuntu@<server-ip>
curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker ubuntu
exit   # log back in so the docker group applies
git clone https://github.com/Makilesh/redline-diligence.git && cd redline-diligence
```

## 5 · Configure

Copy the backend block of `.env.deploy.example` to `.env` in the repository root
and fill in the secrets:

```bash
cp .env.deploy.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # run twice: POSTGRES_PASSWORD, ADMIN_API_KEY
nano .env   # DOMAIN, POSTGRES_PASSWORD, GEMINI_API_KEYS, ADMIN_API_KEY, CORS_ORIGINS
```

`docker compose` refuses to start while `DOMAIN`, `POSTGRES_PASSWORD`,
`ADMIN_API_KEY` or `CORS_ORIGINS` is unset, rather than falling back to a
default password or an unauthenticated admin.

## 6 · Start the stack

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api caddy
```

The first build downloads CPU PyTorch and bakes the embedding, reranker and NLI
weights into the image — 20–30 minutes on 2 ARM cores. The API then needs a few
more minutes to load the models; Caddy waits for its health check. When it is
up:

```bash
curl https://<name>.duckdns.org/health
curl https://<name>.duckdns.org/ready    # 200 once Qdrant and the graph are ready
```

## 7 · Seed the demo data room

Writing to the demo deal is an admin operation. From the server:

```bash
export ADMIN_API_KEY=$(grep ^ADMIN_API_KEY .env | cut -d= -f2-)
for f in data/sample_deal/*.txt; do
  curl -sS -H "X-Admin-Key: $ADMIN_API_KEY" \
       -F "deal_id=aurora_vertex_2024" -F "file=@$f" \
       https://<name>.duckdns.org/api/v1/ingest; echo
done
curl -s https://<name>.duckdns.org/api/v1/deals
```

Nine documents, roughly 110 chunks. Re-running the loop is safe: document and
point IDs are derived from content, so identical files replace themselves
instead of duplicating.

## 8 · Frontend on Vercel, and close the CORS loop

1. Import the repository at <https://vercel.com/new> and set **Root Directory**
   to `web`.
2. Set `NEXT_PUBLIC_API_URL` to `https://<name>.duckdns.org` (no trailing
   slash). It is inlined at **build** time: set it, then redeploy — a build made
   without it calls `http://localhost:8000` from every visitor's browser.
3. Vercel deploys the default branch. Set Settings → Git → Production Branch, or
   merge your work into `main`.
4. Put the Vercel URL in `CORS_ORIGINS` in the server's `.env` and recreate the
   API container:
   `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d api`.
   Get CORS wrong and the failure is silent on the server — the browser blocks
   every request and the UI shows an empty deal list.

Finally, set the repository variable `BACKEND_URL` (Settings → Secrets and
variables → Actions → Variables) to `https://<name>.duckdns.org`; the uptime
workflow in `.github/workflows/keepalive.yml` then checks `/health` every 6
hours.

### Updating

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

The named volumes keep the index, the quota counters and the TLS certificates
across rebuilds.

## 9 · Re-run the eval against the deployed profile

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

The Quality Assessor's relevance floors in `src/agents/quality_assessor.py`
(`RELEVANCE_FLOOR`, `CONFIDENT_REFUSAL_CEILING`, `CONFIDENT_PASS_FLOOR`) were
calibrated against `bge-reranker-v2-m3`'s sigmoid output. The deployment runs
the MiniLM cross-encoder, whose scores are on a comparable but not identical scale,
so if the eval shows more refusals than expected, those floors are the first
place to look — `python -m eval.run_retrieval_eval` reports the gate's decision
per question and runs with `RERANKER_MODEL` set to either model.

---

## What the deployment does not have

**The local Ollama fallback.** The model ladder's last rung is a local
Qwen2.5-14B served by Ollama, which exists to answer when every cloud key is
spent. A 2-core VM cannot run 14B weights alongside the retrieval models at a
usable speed.

The practical effect: on the deployed instance, exhausting the Gemini daily
quota is terminal for that day rather than a quiet downgrade. Since keys
multiply capacity and rotation drains one key per model before stepping down a
rung, the way to push that ceiling out is more keys in `GEMINI_API_KEYS`, not a
different host.

## Not yet verified

- **The image has not been built on ARM.** Every dependency publishes aarch64
  wheels (PyTorch CPU, PyMuPDF, onnxruntime for FastEmbed, psycopg2-binary), but
  `Dockerfile.hf` has not yet been built on an ARM host. The merged compose file
  and the Caddyfile were validated (`docker compose config`, `caddy validate`).
- **Quality Assessor floor recalibration** for the MiniLM reranker — see step 9.
