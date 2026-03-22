#!/usr/bin/env bash
# Simulate a Konflux-style hermetic image build: stage Hermeto output as /cachi2, apply
# project_files overlays, then podman/docker build with --network=none.
#
# Usage:
#   ./scripts/simulate_hermetic_build.sh cpu|cuda
#
# Environment:
#   HERMETO_OUT         Hermeto output dir (default: .hermeto-output)
#   HERMETIC_STAGING    Staging under repo (default: .hermetic-staging)
#   CONTAINER_ENGINE    podman | docker
#   EXTRA_BUILD_ARGS    extra args to build (e.g. --build-arg BASE_IMAGE=registry.access.redhat.com/ubi9/ubi)
#   NETWORK_MODE        unset = engine default (dnf/bundle work on a normal laptop).
#                       Set to "none" for a strict no-network build (needs cached base
#                       layers and Konflux-style RPM prefetch for a cold build).
#   PULL_POLICY         podman/docker --pull value (default: never). Use "missing" or "always"
#                       when the base image is not local yet.
#
# Caveats:
#   - Base image: set PULL_POLICY=missing or always until the tag exists locally; default is never.
#   - Containerfile early RUNs (dnf, pip install uv, bundle install) are unchanged. Use
#     NETWORK_MODE=none only when those layers are already cached or RPMs/gems are prefetched
#     like on Konflux; otherwise omit NETWORK_MODE so dnf/bundle can reach the network.
#   - Hermeto pip-* output does not include deps/generic/model.safetensors; use full-cpu,
#     --model, embeddings_model/, or ALLOW_PLACEHOLDER_HERMETIC_MODEL=1 for a build-only stub.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="${HERMETIC_STAGING:-$ROOT/.hermetic-staging}"
PULL_POLICY="${PULL_POLICY:-never}"

[[ "${1:-}" == cpu || "${1:-}" == cuda ]] || {
	echo "usage: $0 cpu|cuda" >&2
	exit 1
}
kind="$1"

out="${HERMETO_OUT:-$ROOT/.hermeto-output}"
if [[ "$kind" == cuda ]]; then
	if ! python3 -c "import json, pathlib; p=pathlib.Path('$out')/'.build-config.json'; d=json.loads(p.read_text()); names=[pathlib.Path(x['abspath']).name for x in d.get('project_files',[])]; raise SystemExit(0 if any('cuda' in n for n in names) else 1)"; then
		echo "error: HERMETO_OUT ($out) looks like CPU prefetch; run pip-cuda into this directory first" >&2
		exit 1
	fi
fi

engine="${CONTAINER_ENGINE:-}"
if [[ -z "$engine" ]]; then
	if command -v podman >/dev/null 2>&1; then
		engine=podman
	elif command -v docker >/dev/null 2>&1; then
		engine=docker
	else
		echo "error: install podman or docker" >&2
		exit 1
	fi
fi

"$ROOT/scripts/stage_hermetic_build_context.sh" --hermeto-out "${HERMETO_OUT:-$ROOT/.hermeto-output}"

mkdir -p "$STAGING"
gen="$STAGING/Containerfile.sim.$kind"
"$ROOT/scripts/gen_containerfile_hermetic_sim.sh" "$kind" >"$gen"

tag="rag-content-hermetic-sim:$kind"
ctx="$ROOT"

echo "Generated $gen"
echo "Building $tag (network=${NETWORK_MODE:-default}) with $engine..."

build_args=(-f "$gen" -t "$tag" "$ctx" --pull="$PULL_POLICY")
# Only pass --network for an explicit offline build. Podman rootless often breaks on
# `--network=default`; omit the flag to use the engine's normal build networking.
if [[ "${NETWORK_MODE:-}" == none ]]; then
	build_args+=(--network=none)
fi

# shellcheck disable=2206
extra=(${EXTRA_BUILD_ARGS:-})
exec "$engine" build "${extra[@]}" "${build_args[@]}"
