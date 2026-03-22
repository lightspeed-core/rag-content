#!/usr/bin/env bash
# Stage Konflux-style /cachi2 tree and pip requirement overlays from Hermeto output.
#
# Hermeto writes .build-config.json with PIP_FIND_LINKS=${output_dir}/deps/pip and
# project_files templates using file://${output_dir}/deps/pip/.... Konflux substitutes
# output_dir at build time; locally we use /cachi2/output.
#
# Usage:
#   ./scripts/stage_hermetic_build_context.sh [--model PATH] [--hermeto-out DIR]
#
# Model: prefers deps/generic/model.safetensors from Hermeto; else --model; else
# embeddings_model/model.safetensors in the repo; else ALLOW_PLACEHOLDER_HERMETIC_MODEL=1
# (writes a tiny stub so the image build can finish — not runnable for RAG).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMETO_OUT="${HERMETO_OUT:-$ROOT/.hermeto-output}"
STAGING="${HERMETIC_STAGING:-$ROOT/.hermetic-staging}"
MODEL_SRC=""

while [[ "${1:-}" ]]; do
	case "$1" in
	--hermeto-out)
		HERMETO_OUT="$2"
		shift 2
		;;
	--model)
		MODEL_SRC="$2"
		shift 2
		;;
	*)
		echo "usage: $0 [--hermeto-out DIR] [--model PATH]" >&2
		exit 1
		;;
	esac
done

BUILD_CONFIG="$HERMETO_OUT/.build-config.json"
[[ -f "$BUILD_CONFIG" ]] || {
	echo "error: missing $BUILD_CONFIG (run scripts/run_hermeto_fetch_deps.sh pip-cpu|pip-cuda|full-cpu|full-cuda first)" >&2
	exit 1
}
[[ -d "$HERMETO_OUT/deps/pip" ]] || {
	echo "error: missing $HERMETO_OUT/deps/pip" >&2
	exit 1
}

CACHI2="$STAGING/cachi2"
PATCHED="$STAGING/patched-requirements"
rm -rf "$CACHI2" "$PATCHED"
mkdir -p "$CACHI2/output" "$PATCHED"

cp -a "$HERMETO_OUT/deps" "$CACHI2/output/"

mkdir -p "$CACHI2/output/deps/generic"
if [[ -f "$HERMETO_OUT/deps/generic/model.safetensors" ]]; then
	cp -a "$HERMETO_OUT/deps/generic/model.safetensors" "$CACHI2/output/deps/generic/"
elif [[ -n "$MODEL_SRC" ]]; then
	cp -a "$MODEL_SRC" "$CACHI2/output/deps/generic/model.safetensors"
elif [[ -f "$ROOT/embeddings_model/model.safetensors" ]]; then
	cp -a "$ROOT/embeddings_model/model.safetensors" "$CACHI2/output/deps/generic/model.safetensors"
elif [[ "${ALLOW_PLACEHOLDER_HERMETIC_MODEL:-}" == 1 ]]; then
	printf '\0' >"$CACHI2/output/deps/generic/model.safetensors"
	echo "warning: using 1-byte placeholder model (ALLOW_PLACEHOLDER_HERMETIC_MODEL=1)" >&2
else
	echo "error: no model.safetensors (Hermeto generic, --model, embeddings_model/, or set ALLOW_PLACEHOLDER_HERMETIC_MODEL=1)" >&2
	exit 1
fi

# Konflux-style env (Hermeto uses ${output_dir}/deps/pip → /cachi2/output/deps/pip).
cat >"$CACHI2/cachi2.env" <<'EOF'
export PIP_FIND_LINKS=/cachi2/output/deps/pip
export PIP_NO_INDEX=true
EOF

python3 - <<'PY' "$BUILD_CONFIG" "$PATCHED"
import json
import sys
from pathlib import Path

build_config = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
data = json.loads(build_config.read_text())
for pf in data.get("project_files", []):
    path = Path(pf["abspath"])
    name = path.name
    text = pf["template"].replace("${output_dir}", "/cachi2/output")
    (out_dir / name).write_text(text)
    print("patched", name)
PY

echo "Staged: $CACHI2"
echo "Patched requirements: $PATCHED"
