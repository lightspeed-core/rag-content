#!/bin/bash
# Konflux integration test: rag-content → lightspeed-stack E2E validation.
# Runs locally with Podman or inside a Tekton step.
#
# Required env vars:
#   RAG_CONTENT_IMAGE      — just-built rag-content image (from SNAPSHOT)
#   LIGHTSPEED_STACK_IMAGE — lightspeed-stack image (from SNAPSHOT)
#   OPENAI_API_KEY         — OpenAI API key for gpt-4o-mini inference
#
# Optional:
#   GPU_ENABLED            — set to "true" to enable GPU device passthrough and
#                            verify CUDA is available before DB generation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/config"
CORPUS_DIR="$SCRIPT_DIR/corpus"

LIGHTSPEED_STACK_IMAGE="${LIGHTSPEED_STACK_IMAGE:?ERROR: LIGHTSPEED_STACK_IMAGE not set}"

RAG_OUTPUT_DIR=$(mktemp -d /tmp/rag-integration-output.XXXXXX)
MODEL_DIR=$(mktemp -d /tmp/rag-integration-model.XXXXXX)

progress() { echo "[rag-e2e] $*"; }

cleanup() {
  progress "Cleaning up..."
  podman stop lightspeed-stack-e2e 2>/dev/null || true
  podman rm lightspeed-stack-e2e 2>/dev/null || true
  podman unshare rm -rf "$RAG_OUTPUT_DIR" "$MODEL_DIR" 2>/dev/null || rm -rf "$RAG_OUTPUT_DIR" "$MODEL_DIR" 2>/dev/null || true
}
trap cleanup EXIT

[[ -n "${RAG_CONTENT_IMAGE:-}" ]] || { echo "ERROR: RAG_CONTENT_IMAGE not set"; exit 1; }
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: OPENAI_API_KEY not set"; exit 1; }

progress "RAG_CONTENT_IMAGE=$RAG_CONTENT_IMAGE"
progress "LIGHTSPEED_STACK_IMAGE=$LIGHTSPEED_STACK_IMAGE"
progress "GPU_ENABLED=${GPU_ENABLED:-false}"

GPU_FLAGS=()
if [ "${GPU_ENABLED:-}" = "true" ]; then
  GPU_FLAGS=(--device nvidia.com/gpu=all --security-opt=label=disable)

  progress "GPU pre-check: verifying CUDA is available in rag-content image..."
  podman run --rm --network=host "${GPU_FLAGS[@]}" \
    "$RAG_CONTENT_IMAGE" \
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — GPU device not accessible'; print(f'GPU verified: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}')"
  progress "GPU pre-check passed"
fi

#========================================
# Phase 1: Generate FAISS DB
#========================================
progress "Phase 1/6: Generating FAISS vector DB from test corpus..."
chmod 777 "$RAG_OUTPUT_DIR"
podman run --rm --network=host "${GPU_FLAGS[@]}" \
  -v "$CORPUS_DIR":/input:ro \
  -v "$RAG_OUTPUT_DIR":/output \
  "$RAG_CONTENT_IMAGE" \
  python /rag-content/scripts/generate_embeddings.py \
    -f /input -o /output -i e2e-test-index \
    -s llamastack-faiss \
    -d /rag-content/embeddings_model

podman unshare chmod -R 777 "$RAG_OUTPUT_DIR" 2>/dev/null || chmod -R 777 "$RAG_OUTPUT_DIR"
progress "DB generation complete. Output files:"
ls -la "$RAG_OUTPUT_DIR"

#========================================
# Phase 2: Copy embedding model
#========================================
progress "Phase 2/6: Copying embedding model from rag-content image..."
chmod 777 "$MODEL_DIR"
podman run --rm --network=host "${GPU_FLAGS[@]}" \
  -v "$MODEL_DIR":/out \
  "$RAG_CONTENT_IMAGE" \
  bash -c "cp -r /rag-content/embeddings_model/. /out/"

podman unshare chmod -R 777 "$MODEL_DIR" 2>/dev/null || chmod -R 777 "$MODEL_DIR"
progress "Embedding model copied. Files:"
ls -la "$MODEL_DIR"

#========================================
# Phase 3: Extract vector store ID
#========================================
progress "Phase 3/6: Extracting FAISS vector store ID..."
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

[[ -n "$FAISS_VECTOR_STORE_ID" ]] || { echo "ERROR: Failed to extract FAISS_VECTOR_STORE_ID"; exit 1; }
progress "FAISS_VECTOR_STORE_ID=$FAISS_VECTOR_STORE_ID"

#========================================
# Phase 4: Start lightspeed-stack
#========================================
progress "Phase 4/6: Starting lightspeed-stack (library mode)..."
chmod -R a+r "$CONFIG_DIR"
podman run -d --name lightspeed-stack-e2e \
  --network=host \
  --security-opt label=disable \
  -v "$RAG_OUTPUT_DIR":/opt/app-root/src/.llama/storage/rag \
  -v "$MODEL_DIR":/embeddings \
  -v "$CONFIG_DIR/lightspeed-stack.yaml":/app-root/lightspeed-stack.yaml:ro \
  -v "$CONFIG_DIR/run.yaml":/app-root/run.yaml:ro \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e FAISS_VECTOR_STORE_ID="$FAISS_VECTOR_STORE_ID" \
  -e HF_HOME=/embeddings \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  "$LIGHTSPEED_STACK_IMAGE"

#========================================
# Phase 5: Wait for healthy
#========================================
progress "Phase 5/6: Waiting for lightspeed-stack to become healthy..."
for i in $(seq 1 60); do
  if ! podman ps --format '{{.Names}}' | grep -q lightspeed-stack-e2e; then
    progress "ERROR: Container exited unexpectedly"
    progress "Container logs:"
    podman logs lightspeed-stack-e2e 2>&1 | tail -100
    exit 1
  fi
  HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/liveness 2>/dev/null || echo "000")
  if [ "$HTTP_CODE" = "200" ]; then
    progress "Lightspeed-stack healthy after $(( i * 5 ))s (HTTP $HTTP_CODE)"
    break
  fi
  if [ "$i" -eq 60 ]; then
    progress "ERROR: Lightspeed-stack not healthy after 300s (last HTTP code: $HTTP_CODE)"
    progress "Container logs:"
    podman logs lightspeed-stack-e2e 2>&1 | tail -200
    exit 1
  fi
  sleep 5
done

#========================================
# Phase 6: Query and assert
#========================================
progress "Phase 6/6: Sending query and validating RAG response..."
progress "Lightspeed-stack logs before query:"
podman logs lightspeed-stack-e2e 2>&1 | tail -30
QUERY_START=$(date +%s)
CURL_HEADERS=$(mktemp)
HTTP_RESPONSE=$(curl -s -D "$CURL_HEADERS" --max-time 120 -w "\n%{http_code}" -X POST http://localhost:8080/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Zyranex Model T7 and what company makes it?",
    "system_prompt": "You are an assistant. Always use the file_search tool to answer.",
    "model": "gpt-4o-mini",
    "provider": "openai"
  }') || true
progress "Response headers:"
cat "$CURL_HEADERS"
rm -f "$CURL_HEADERS"
QUERY_HTTP_CODE=$(echo "$HTTP_RESPONSE" | tail -1)
RESPONSE=$(echo "$HTTP_RESPONSE" | sed '$d')

QUERY_END=$(date +%s)
progress "Query took $(( QUERY_END - QUERY_START ))s, returned HTTP $QUERY_HTTP_CODE"
progress "Lightspeed-stack logs after query:"
podman logs lightspeed-stack-e2e 2>&1 | tail -30
progress "Full response:"
echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"

if [ "$QUERY_HTTP_CODE" != "200" ]; then
  progress "FAILURE: Query returned HTTP $QUERY_HTTP_CODE"
  progress "Full response:"
  echo "$RESPONSE"
  progress "Container logs:"
  podman logs lightspeed-stack-e2e 2>&1 | tail -100
  exit 1
fi

if echo "$RESPONSE" | grep -qi "zyranex\|quorbitex\|ZRX-4401"; then
  progress "SUCCESS: RAG response contains expected test corpus terms"
  progress "Full lightspeed-stack logs:"
  podman logs lightspeed-stack-e2e 2>&1 | tail -50
  exit 0
else
  progress "FAILURE: RAG response does not contain expected test corpus terms"
  progress "Full response:"
  echo "$RESPONSE"
  progress "Container logs:"
  podman logs lightspeed-stack-e2e 2>&1 | tail -100
  exit 1
fi
