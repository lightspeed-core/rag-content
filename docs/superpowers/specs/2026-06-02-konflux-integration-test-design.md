# Konflux Integration Test: rag-content → lightspeed-stack E2E Validation

**Status:** Implemented
**Date:** 2026-06-02
**Author:** AI-assisted design

## Problem

There is no automated integration test validating the full path from rag-content
index generation through to lightspeed-stack consumption. A breaking change in
either repo (schema drift, embedding format change, metadata structure) goes
undetected until manual testing or production.

## Goal

Add a Konflux integration test pipeline to the rag-content repository that runs
after every successful image build and validates the generated FAISS vector DB
works end-to-end with lightspeed-stack, on both x86_64 and arm64.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Infrastructure | Multi-platform VMs via Konflux multi-platform controller | Native multi-arch testing on x86_64 and arm64; same infrastructure as build pipelines |
| Pipeline pattern | Tekton matrix fan-out → VM SSH → shell script | Mirrors ramalama's multi-arch integration test pattern; locally reproducible |
| Lightspeed-stack deployment | Library mode (embedded llama-stack) via Podman on VM | Single container; no separate llama-stack service |
| Lightspeed-stack image | `registry.redhat.io/lightspeed-core/lightspeed-stack-rhel9:v0.6.0rc2` (pinned) | Deterministic; bump manually as needed |
| LLM provider | OpenAI (gpt-4o-mini) via Konflux secret | Same pattern as lightspeed-stack E2E |
| Test corpus | Synthetic fictional product manual with YAML frontmatter | Unique terms impossible in LLM training data; valid URL for lightspeed-stack citations |
| Assertion method | grep on curl response for distinctive corpus terms | Simple, no test framework needed |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Tekton Pipeline: rag-content-integration-tests-pipeline         │
│                                                                  │
│  Task: get-rag-content-image                                     │
│    Extract image, commit, repo URL from SNAPSHOT                 │
│                                                                  │
│  Task: run-integration-test (matrix fan-out)                     │
│    ┌──────────────────────┐  ┌──────────────────────┐            │
│    │  linux/x86_64 VM     │  │  linux-c6gd2xlarge/   │            │
│    │                      │  │  arm64 VM             │            │
│    │  test-vm-cmd task:   │  │  test-vm-cmd task:    │            │
│    │  SSH into VM →       │  │  SSH into VM →        │            │
│    │  pipeline-konflux.sh │  │  pipeline-konflux.sh  │            │
│    └──────────────────────┘  └──────────────────────┘            │
│                                                                  │
│  pipeline-konflux.sh (runs on each VM via Podman):               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  1. GENERATE DB                                            │  │
│  │     podman run $RAG_CONTENT_IMAGE generate_embeddings.py   │  │
│  │                                                            │  │
│  │  2. COPY EMBEDDING MODEL                                   │  │
│  │     podman run $RAG_CONTENT_IMAGE cp embeddings_model      │  │
│  │                                                            │  │
│  │  3. EXTRACT FAISS_VECTOR_STORE_ID                          │  │
│  │     python3 sqlite3 query on faiss_store.db                │  │
│  │                                                            │  │
│  │  4. SERVE                                                  │  │
│  │     podman run -d --network=host lightspeed-stack           │  │
│  │     (library mode, configs at /app-root/)                  │  │
│  │                                                            │  │
│  │  5. WAIT HEALTHY                                           │  │
│  │     poll localhost:8080/liveness (timeout 300s)             │  │
│  │                                                            │  │
│  │  6. QUERY + ASSERT                                         │  │
│  │     curl POST /v1/query, check HTTP 200                    │  │
│  │     grep for "Zyranex" / "Quorbitex" / "ZRX-4401"         │  │
│  │                                                            │  │
│  │  7. TEARDOWN (trap EXIT)                                   │  │
│  │     podman stop/rm, podman unshare rm temp dirs            │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## File Layout

```
.tekton/
  integration-tests/
    pipeline/
      rag-content-integration-test.yaml
    tasks/
      test-vm-cmd.yaml

tests/
  integration-konflux/
    pipeline-konflux.sh
    config/
      lightspeed-stack.yaml
      run.yaml
    corpus/
      manual.md
```

### `.tekton/integration-tests/pipeline/rag-content-integration-test.yaml`

Tekton Pipeline with multi-arch matrix fan-out.

**Params:**
- `SNAPSHOT` — JSON string with the built rag-content image (provided by Konflux)
- `test-name` — defaults to `rag-content-e2e-tests`
- `platforms` — VM platforms, defaults to `[linux/x86_64, linux-c6gd2xlarge/arm64]`

**Tasks:**
1. `get-rag-content-image` — parse SNAPSHOT JSON with `jq` to extract the
   rag-content container image URL, git revision, and repo URL.
2. `echo-integration-params` — log parameters for debugging.
3. `run-integration-test` — matrix fan-out across platforms, each using
   `test-vm-cmd` task to SSH into a provisioned VM and run the test script.

### `.tekton/integration-tests/tasks/test-vm-cmd.yaml`

Tekton Task that runs a command on a multi-platform VM provisioned by the
Konflux multi-platform controller. Adapted from `containers/ramalama`'s
multi-arch integration test pattern.

- Reads SSH credentials from `multi-platform-ssh-$(context.taskRun.name)` secret
- Reads OpenAI API key from `openai-api-key` secret
- Injects environment variables and secrets into a script
- RSyncs the script to the VM and executes it via SSH
- Produces standardized `TEST_OUTPUT` result

### `tests/integration-konflux/pipeline-konflux.sh`

Main orchestration script. Designed to be runnable locally with Podman for
development/debugging. Uses `mktemp` for unique temporary directories.

**Phase 1 — Generate FAISS DB:**

```bash
RAG_OUTPUT_DIR=$(mktemp -d /tmp/rag-integration-output.XXXXXX)
chmod 777 "$RAG_OUTPUT_DIR"
podman run --rm \
  -v "$CORPUS_DIR":/input:ro,Z \
  -v "$RAG_OUTPUT_DIR":/output:Z \
  "$RAG_CONTENT_IMAGE" \
  python /rag-content/scripts/generate_embeddings.py \
    -f /input -o /output -i e2e-test-index \
    -s llamastack-faiss \
    -d /rag-content/embeddings_model
podman unshare chmod -R 777 "$RAG_OUTPUT_DIR"
```

Runs the just-built rag-content image to produce a FAISS vector DB. Output
directory is `chmod 777` before and after for cross-uid access (rag-content
runs as uid 1000, lightspeed-stack as uid 1001). `podman unshare` handles
uid-mapped files from rootless podman.

**Phase 2 — Copy embedding model:**

```bash
MODEL_DIR=$(mktemp -d /tmp/rag-integration-model.XXXXXX)
chmod 777 "$MODEL_DIR"
podman run --rm \
  -v "$MODEL_DIR":/out:Z \
  "$RAG_CONTENT_IMAGE" \
  bash -c "cp -r /rag-content/embeddings_model/. /out/"
podman unshare chmod -R 777 "$MODEL_DIR"
```

Copies the sentence-transformers model from the rag-content image so
lightspeed-stack can load it locally via `provider_model_id: /embeddings`.

**Phase 3 — Extract vector store ID:**

```bash
FAISS_VECTOR_STORE_ID=$(python3 -c "
import sqlite3, re, sys
conn = sqlite3.connect('$RAG_OUTPUT_DIR/faiss_store.db')
cursor = conn.cursor()
cursor.execute(\"SELECT key FROM kvstore WHERE key LIKE '%vector_stores:v%::%' LIMIT 1\")
row = cursor.fetchone()
if row:
    match = re.search(r'(vs_[a-f0-9-]+)', row[0])
    if match:
        print(match.group(1))
        sys.exit(0)
print('ERROR: no vector store found', file=sys.stderr)
sys.exit(1)
")
```

Same extraction pattern as lightspeed-stack's `pipeline-konflux.sh`.

**Phase 4 — Start lightspeed-stack:**

```bash
podman run -d --name lightspeed-stack-e2e \
  --network=host \
  -v "$RAG_OUTPUT_DIR":/opt/app-root/src/.llama/storage/rag:Z \
  -v "$MODEL_DIR":/embeddings:Z \
  -v "$CONFIG_DIR/lightspeed-stack.yaml":/app-root/lightspeed-stack.yaml:ro,Z \
  -v "$CONFIG_DIR/run.yaml":/app-root/run.yaml:ro,Z \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e FAISS_VECTOR_STORE_ID="$FAISS_VECTOR_STORE_ID" \
  -e HF_HOME=/embeddings \
  "$LIGHTSPEED_STACK_IMAGE"
```

Key details (learned from lightspeed-stack's `docker-compose-library.yaml`):
- `--network=host` for reliable localhost access
- DB mounted at `/opt/app-root/src/.llama/storage/rag` (standard llama path)
- DB volume is read-write (llama-stack writes during vector store registration)
- Configs at `/app-root/` (container workdir, default config path)
- Model at `/embeddings` (referenced by `run.yaml` `provider_model_id`)

**Phase 5 — Wait for healthy:**

Poll `http://localhost:8080/liveness` every 5 seconds, timeout after 300
seconds. Detects container crashes early. Shows HTTP status code for debugging.

**Phase 6 — Query and assert:**

```bash
HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Zyranex Model T7 and what company makes it?",
    "system_prompt": "You are an assistant. Always use the file_search tool to answer.",
    "model": "gpt-4o-mini",
    "provider": "openai"
  }') || true
QUERY_HTTP_CODE=$(echo "$HTTP_RESPONSE" | tail -1)
RESPONSE=$(echo "$HTTP_RESPONSE" | sed '$d')
```

Captures HTTP code separately (avoids `set -e` abort on errors). Checks for
HTTP 200 first, then greps for distinctive corpus terms.

**Phase 7 — Teardown (trap EXIT):**

```bash
cleanup() {
  podman stop lightspeed-stack-e2e 2>/dev/null || true
  podman rm lightspeed-stack-e2e 2>/dev/null || true
  podman unshare rm -rf "$RAG_OUTPUT_DIR" "$MODEL_DIR" 2>/dev/null || \
    rm -rf "$RAG_OUTPUT_DIR" "$MODEL_DIR" 2>/dev/null || true
}
trap cleanup EXIT
```

Uses `podman unshare` to remove uid-mapped files from rootless podman.

### `tests/integration-konflux/config/lightspeed-stack.yaml`

Adapted from `lightspeed-stack/tests/e2e/configuration/library-mode/lightspeed-stack.yaml`
and `lightspeed-stack/docker-compose-library.yaml`:

```yaml
name: Lightspeed Core Service (LCS) — rag-content integration test
service:
  host: 0.0.0.0
  port: 8080
  auth_enabled: false
  workers: 1
  color_log: true
  access_log: true
llama_stack:
  use_as_library_client: true
  library_client_config_path: run.yaml
user_data_collection:
  feedback_enabled: false
  transcripts_enabled: false
authentication:
  module: "noop"
inference:
  default_provider: openai
  default_model: gpt-4o-mini
byok_rag:
  - rag_id: e2e-test-docs
    rag_type: inline::faiss
    embedding_model: sentence-transformers/all-mpnet-base-v2
    embedding_dimension: 768
    vector_db_id: ${env.FAISS_VECTOR_STORE_ID}
    db_path: /opt/app-root/src/.llama/storage/rag/faiss_store.db
    score_multiplier: 1.0
rag:
  tool:
    - e2e-test-docs
```

### `tests/integration-konflux/config/run.yaml`

Based on `lightspeed-stack/tests/e2e/configs/run-ci.yaml`. Includes all
providers required by the `/v1/query` endpoint:

- `remote::openai` inference (reads `OPENAI_API_KEY` from env)
- `inline::sentence-transformers` embeddings (model at `/embeddings`)
- `inline::rag-runtime` tool provider
- `inline::localfs` files provider
- `inline::llama-guard` safety provider (required for shield moderation)
- `inline::meta-reference` agents provider (required for responses API)
- `kv_sqlite` + `sql_sqlite` storage backends
- Registered `all-mpnet-base-v2` model, `llama-guard` shield, `builtin::rag` tool group

### `tests/integration-konflux/corpus/manual.md`

A short synthetic product manual (~60 lines) with YAML frontmatter containing
a valid URL (required by lightspeed-stack's `ReferencedDocument` pydantic model
for citation URLs). Contains distinctive invented terms:

- **Company:** Quorbitex Industries
- **Product:** Zyranex Model T7 (plasma-cooled quantum relay)
- **Error codes:** ZRX-4401, ZRX-4402, ZRX-7710
- **Proprietary terms:** sub-orbital mesh networking, Quorbitex Reliability
  Protocol (QRP), Phase-Lock Calibration Sequence

### Standardized TEST_OUTPUT result

Per the Konflux integration test contract, each matrix task produces a
`TEST_OUTPUT` Tekton result in JSON format:

```json
{"result":"SUCCESS","timestamp":"2026-06-02T12:00:00+00:00","successes":1,"failures":0,"warnings":0}
```

### Secrets required in Konflux namespace

| Secret | Key | Source |
|--------|-----|--------|
| `openai-api-key` | `openai-api-key` | OpenAI API key for gpt-4o-mini inference |

The `multi-platform-ssh-*` secrets are auto-created by the Konflux
multi-platform controller.

### Component name in SNAPSHOT

The existing `.tekton/rag-tool-push.yaml` uses `rag-tool` as the component name
in its labels. The SNAPSHOT JSON will use this name, so we extract the image with
`select(.name == "rag-tool")`.

## What This Validates

The integration test validates the full pipeline from index generation to serving:

1. **Image build correctness** — the rag-content image can run
   `generate_embeddings.py` and produce valid output
2. **Embedding model bundling** — the model at `/rag-content/embeddings_model/`
   is present, loadable, and produces valid embeddings
3. **FAISS DB format** — the generated `faiss_store.db` schema
   is compatible with llama-stack's `inline::faiss` provider
4. **byok_rag integration** — lightspeed-stack can configure and load the DB via
   `byok_rag` config
5. **End-to-end RAG retrieval** — a query retrieves context from the test corpus
   and includes it in the LLM-augmented response
6. **Multi-arch compatibility** — all of the above works on both x86_64 and arm64

## What This Does NOT Validate

- GPU/CUDA code paths (test uses CPU image only)
- Postgres vector store backends
- Multi-product document processing or OKP filtering
- Lightspeed-stack features unrelated to RAG (auth, MCP, streaming, etc.)
- Performance or scale (single small corpus, single query)

## Failure Modes and Debugging

| Failure | Signal | Debug |
|---------|--------|-------|
| DB generation fails | `podman run` exits non-zero | Check container logs; verify corpus format |
| Missing vector store ID | Empty `FAISS_VECTOR_STORE_ID` | Inspect `faiss_store.db` sqlite keys |
| Lightspeed-stack won't start | Health check timeout | `podman logs lightspeed-stack-e2e`; check config paths |
| DB write error at startup | `sqlite3.OperationalError: readonly` | Verify DB volume is mounted read-write, permissions are 777 |
| Query returns HTTP 500 | Missing API/provider | Check `run.yaml` has all required APIs (agents, safety) |
| Query returns no RAG context | grep finds no matching terms | Check `byok_rag` config, DB mount path, vector store ID |
| OpenAI API error | curl returns 5xx/4xx | Verify secret is mounted and API key is valid |
| URL validation error | `doc_url` pydantic error | Ensure corpus has YAML frontmatter with valid `url` field |

## Local Development

Developers can run the test locally with Podman:

```bash
export RAG_CONTENT_IMAGE=localhost/rag-content:cpu
export OPENAI_API_KEY=sk-...
cd tests/integration-konflux
./pipeline-konflux.sh
```

Build the local rag-content image first with:
```bash
podman build -t localhost/rag-content:cpu -f Containerfile .
```
