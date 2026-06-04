# Konflux Integration Test Implementation Plan

> **Status:** Implemented. This plan documents the implemented tasks for reference.

**Goal:** Add a Konflux integration test that validates rag-content FAISS vector DB generation works end-to-end with lightspeed-stack on both x86_64 and arm64, for both CPU (`rag-tool`) and CUDA (`rag-tool-cuda`) image variants (2 platforms × 2 images = 4 matrix runs).

**Architecture:** Tekton Pipeline with 2×2 matrix fan-out across platforms and images. The `init-snapshot` task extracts separate `cpu-image` and `cuda-image` from the SNAPSHOT JSON. The multi-platform controller provisions VMs for each arch. The task spec is inlined in the pipeline YAML — it SSHs into each VM, clones the repo in the Tekton step, rsyncs to the VM, then runs `pipeline-konflux.sh` inside a privileged container (`--privileged --network=host --security-opt label=disable --security-opt seccomp=unconfined -e STORAGE_DRIVER=vfs`). The script uses Podman to run the built rag-content image (DB generation) and `quay.io/lightspeed-core/lightspeed-stack:dev-latest` (library mode, serving), then queries and asserts RAG context from a synthetic corpus. `test-vm-cmd.yaml` exists as a standalone reference but is not used by the pipeline.

**Tech Stack:** Tekton/Konflux pipelines, Konflux multi-platform controller, Podman, Bash, lightspeed-stack (library mode), OpenAI API, FAISS/llama-stack

**Spec:** `docs/superpowers/specs/2026-06-02-konflux-integration-test-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `.tekton/integration-tests/pipeline/rag-content-integration-test.yaml` | Tekton Pipeline — SNAPSHOT extraction (cpu-image + cuda-image), matrix fan-out across platforms × images |
| `.tekton/integration-tests/pipeline/its.yaml` | IntegrationTestScenario — registers the pipeline with Konflux |
| `.tekton/integration-tests/tasks/test-vm-cmd.yaml` | Standalone reference Tekton Task (not used by pipeline — task spec is inlined) |
| `tests/integration-konflux/pipeline-konflux.sh` | Main orchestration — DB generation, serving, query, assertion, teardown |
| `tests/integration-konflux/config/lightspeed-stack.yaml` | Lightspeed-stack config — library mode, byok_rag, noop auth |
| `tests/integration-konflux/config/run.yaml` | Llama-stack config — OpenAI, sentence-transformers, safety, agents, RAG |
| `tests/integration-konflux/corpus/manual.md` | Synthetic test corpus — fictional product manual with YAML frontmatter |

---

### Task 1: Create the synthetic test corpus

**Files:** `tests/integration-konflux/corpus/manual.md`

Synthetic Quorbitex/Zyranex product manual with YAML frontmatter containing a
valid URL (required by lightspeed-stack's `ReferencedDocument` model for
citation URLs). Content uses distinctive invented terms for assertion.

### Task 2: Create the lightspeed-stack configuration

**Files:** `tests/integration-konflux/config/lightspeed-stack.yaml`

Library mode config adapted from `lightspeed-stack/docker-compose-library.yaml`:
- `use_as_library_client: true` with `library_client_config_path: run.yaml` (relative path, resolves in `/app-root/`)
- `user_data_collection` with feedback/transcripts disabled (required field)
- `byok_rag` with `db_path` pointing to the standard llama storage path
- `authentication: noop`

### Task 3: Create the llama-stack run.yaml

**Files:** `tests/integration-konflux/config/run.yaml`

Based on `lightspeed-stack/tests/e2e/configs/run-ci.yaml` with all providers
required by the `/v1/query` endpoint:
- `agents` API + `inline::meta-reference` (required by responses API)
- `safety` API + `inline::llama-guard` + shield registration (required for shield moderation)
- `inference` with `remote::openai` + `inline::sentence-transformers`
- `files`, `tool_runtime` (`inline::rag-runtime`), `vector_io`
- `conversations` store (required by agents provider)
- Embedding model `provider_model_id: /embeddings` (local mount path)

### Task 4: Create the pipeline shell script

**Files:** `tests/integration-konflux/pipeline-konflux.sh` (executable)

6-phase orchestration with Podman. Key implementation details:
- `mktemp -d` for unique temporary directories
- `chmod 777` + `podman unshare chmod` for cross-uid access (rag-content=uid 1000, lightspeed-stack=uid 1001)
- `--network=host` on all `podman run` calls for reliable localhost access
- No `:Z` SELinux labels on volume mounts (removed for nested podman compatibility)
- DB mounted at `/opt/app-root/src/.llama/storage/rag` (standard llama path from docker-compose)
- DB volume read-write (llama-stack writes during vector store registration)
- Health check at `/liveness` (not `/v1/liveness` or `/v1/models`)
- `curl --max-time 120` for the query request
- Query captures HTTP code separately to avoid `set -e` abort
- Cleanup uses `podman unshare rm` for uid-mapped files
- `HF_TOKEN` passed through conditionally via `${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"}`
- Lightspeed-stack image defaults to `quay.io/lightspeed-core/lightspeed-stack:dev-latest`

### Task 5: Create the Tekton Pipeline YAML

**Files:**
- `.tekton/integration-tests/pipeline/rag-content-integration-test.yaml`
- `.tekton/integration-tests/pipeline/its.yaml`

Pipeline uses Tekton `matrix` to fan out `run-integration-test` across
platforms (`linux-mlarge/amd64`, `linux-mlarge/arm64`) × images (cpu-image,
cuda-image) = 4 runs. Task spec is inlined in the pipeline (no taskRef).

The `init-snapshot` task extracts two separate images from the SNAPSHOT:
- `cpu-image` from the `rag-tool` component
- `cuda-image` from the `rag-tool-cuda` component

Each matrix cell:
- Multi-platform controller provisions a VM and creates SSH credentials
- Tekton step clones the repo, rsyncs to VM
- Runs `pipeline-konflux.sh` inside a privileged container on the VM with
  `--privileged --network=host --security-opt label=disable --security-opt seccomp=unconfined -e STORAGE_DRIVER=vfs`
- Injects `OPENAI_API_KEY` (from `openai-api-key` secret) and `HF_TOKEN`
  (from `huggingface-token` secret, key `hf-token-ces-lcore-test`)
- Produces standardized `TEST_OUTPUT` result per Konflux contract

### Task 6: Local smoke test

Validated end-to-end locally with `localhost/rag-content:cpu` image.
Issues found and fixed during testing:
- Directory permissions for non-root container users
- Config mount paths (`/app-root/` workdir, relative `run.yaml` path)
- Required `user_data_collection` field
- DB volume must be read-write for llama-stack registration
- `--network=host` needed for rootless podman
- Health check endpoint is `/liveness`
- Query response capture without `set -e` abort
- `agents` and `safety` APIs required by `/v1/query`
- Embedding model `provider_model_id` must point to local mount path
- Corpus needs YAML frontmatter with valid URL for citation parsing

### Task 7: Multi-arch and dual-image support

Added Tekton 2×2 matrix fan-out: platforms (`linux-mlarge/amd64`,
`linux-mlarge/arm64`) × images (cpu-image, cuda-image). The `init-snapshot` task
extracts both `rag-tool` and `rag-tool-cuda` component images from the SNAPSHOT.
The `test-vm-cmd.yaml` standalone task is kept as reference but the pipeline
inlines its own task spec with git clone, rsync, privileged container execution,
and additional secrets (`huggingface-token`).
