#!/usr/bin/env bash
# Run Hermeto like Konflux prefetch-dependencies (see pipeline log):
#   hermeto --log-level debug --mode strict fetch-deps '<json>' --sbom-output-type spdx --source ... --output ...
# Image: https://github.com/hermetoproject/hermeto  (ghcr.io/hermetoproject/hermeto)
#
# Usage:
#   ./scripts/run_hermeto_fetch_deps.sh pip-cpu    # pip input only (matches Konflux pip slice)
#   ./scripts/run_hermeto_fetch_deps.sh pip-cuda
#   ./scripts/run_hermeto_fetch_deps.sh full-cpu # generic + rpm + pip + bundler (no RHSM certs — rpm may fail locally)
#   ./scripts/run_hermeto_fetch_deps.sh full-cuda
#
# Environment:
#   HERMETO_IMAGE      default ghcr.io/hermetoproject/hermeto:latest
#   HERMETO_OUT        output dir (default: <repo>/.hermeto-output)
#   CONTAINER_ENGINE   podman | docker
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMETO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/hermeto" && pwd)"
HERMETO_IMAGE="${HERMETO_IMAGE:-ghcr.io/hermetoproject/hermeto:latest}"
OUT="${HERMETO_OUT:-$ROOT/.hermeto-output}"

usage() {
	echo "usage: $0 {pip-cpu|pip-cuda|full-cpu|full-cuda}" >&2
	exit 1
}

[[ "${1:-}" ]] || usage

engine="${CONTAINER_ENGINE:-}"
if [[ -z "$engine" ]]; then
	if command -v podman >/dev/null 2>&1; then
		engine=podman
	elif command -v docker >/dev/null 2>&1; then
		engine=docker
	else
		echo "error: install podman or docker, or set CONTAINER_ENGINE" >&2
		exit 1
	fi
fi

case "$1" in
pip-cpu) json="$HERMETO_DIR/prefetch-pip-cpu.json" ;;
pip-cuda) json="$HERMETO_DIR/prefetch-pip-cuda.json" ;;
full-cpu) json="$HERMETO_DIR/prefetch-konflux-cpu.json" ;;
full-cuda) json="$HERMETO_DIR/prefetch-konflux-cuda.json" ;;
*) usage ;;
esac

[[ -f "$json" ]] || { echo "error: missing $json" >&2; exit 1; }

mkdir -p "$OUT"

vol_mount=(-v "$ROOT:$ROOT")
if [[ "$engine" == podman ]]; then
	vol_mount=(-v "$ROOT:$ROOT:z")
fi

input_json="$(cat -- "$json")"
if [[ -z "${input_json//[$'\t\n\r ']/}" ]]; then
	echo "error: empty JSON in $json" >&2
	exit 1
fi

echo "Running (Konflux-shaped): $engine run ... $HERMETO_IMAGE \\"
echo "  --log-level debug --mode strict fetch-deps ... --sbom-output-type spdx --source $ROOT --output $OUT"
echo "  mode=$1"

# JSON is the fetch-deps positional argument; options follow (same order as Konflux HermetoCli log).
exec "$engine" run --rm \
	"${vol_mount[@]}" \
	-w "$ROOT" \
	"$HERMETO_IMAGE" \
	--log-level debug \
	--mode strict \
	fetch-deps \
	"$input_json" \
	--sbom-output-type spdx \
	--source "$ROOT" \
	--output "$OUT"
