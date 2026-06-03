# Konflux Integration Test Implementation Plan

> **Status:** Implemented. This plan documents the implemented tasks for reference.

**Goal:** Add a Konflux integration test that validates rag-content FAISS vector DB generation works end-to-end with lightspeed-stack on both x86_64 and arm64.

**Architecture:** Tekton Pipeline with matrix fan-out across platforms. The multi-platform controller provisions VMs for each arch. A `test-vm-cmd` task SSHs into each VM and runs `pipeline-konflux.sh`, which uses Podman to run the built rag-content image (DB generation) and a pinned lightspeed-stack image (library mode, serving), then queries and asserts RAG context from a synthetic corpus.

**Tech Stack:** Tekton/Konflux pipelines, Konflux multi-platform controller, Podman, Bash, lightspeed-stack (library mode), OpenAI API, FAISS/llama-stack

**Spec:** `docs/superpowers/specs/2026-06-02-konflux-integration-test-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `.tekton/integration-tests/pipeline/rag-content-integration-test.yaml` | Tekton Pipeline — SNAPSHOT extraction, matrix fan-out across platforms |
| `.tekton/integration-tests/tasks/test-vm-cmd.yaml` | Tekton Task — SSH into multi-platform VM, inject env vars + secrets, run command |
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

6-phase orchestration with Podman. Key implementation details discovered
during smoke testing:
- `mktemp -d` for unique temporary directories
- `chmod 777` + `podman unshare chmod` for cross-uid access (rag-content=uid 1000, lightspeed-stack=uid 1001)
- `--network=host` for reliable localhost access (rootless podman port forwarding is unreliable)
- DB mounted at `/opt/app-root/src/.llama/storage/rag` (standard llama path from docker-compose)
- DB volume read-write (llama-stack writes during vector store registration)
- Health check at `/liveness` (not `/v1/liveness` or `/v1/models`)
- Query captures HTTP code separately to avoid `set -e` abort
- Cleanup uses `podman unshare rm` for uid-mapped files

### Task 5: Create the Tekton Pipeline and Task YAMLs

**Files:**
- `.tekton/integration-tests/pipeline/rag-content-integration-test.yaml`
- `.tekton/integration-tests/tasks/test-vm-cmd.yaml`

Pipeline uses Tekton `matrix` to fan out `run-integration-test` across
platforms (`linux/x86_64`, `linux-c6gd2xlarge/arm64`). Each platform instance
uses `test-vm-cmd` task (adapted from `containers/ramalama`'s pattern):
- Multi-platform controller provisions a VM and creates SSH credentials
- Task SSHs into the VM, injects env vars and OpenAI API key secret
- Runs a script that installs deps, clones repo, and executes `pipeline-konflux.sh`
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

### Task 7: Multi-arch support

Added Tekton matrix fan-out and `test-vm-cmd` task following the
`containers/ramalama` multi-arch integration test pattern. Platform labels
match the build pipeline: `linux/x86_64` and `linux-c6gd2xlarge/arm64`.
