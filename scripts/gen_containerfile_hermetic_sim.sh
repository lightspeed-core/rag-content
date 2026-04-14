#!/usr/bin/env bash
# Generate a Containerfile that injects staged /cachi2 and Hermeto-patched requirement fragments.
#
# Usage:
#   ./scripts/gen_containerfile_hermetic_sim.sh cpu|cuda > .hermetic-staging/Containerfile.sim
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Paths must be relative to build context (repo root); keep staging under the repo.
STAGING_REL=".hermetic-staging"

[[ "${1:-}" == cpu || "${1:-}" == cuda ]] || {
	echo "usage: $0 cpu|cuda" >&2
	exit 1
}

if [[ "$1" == cpu ]]; then
	base="$ROOT/Containerfile"
else
	base="$ROOT/Containerfile-cuda"
fi

awk -v staging="$STAGING_REL" '
/^USER root$/ {
	print
	print ""
	print "# Injected by gen_containerfile_hermetic_sim.sh (local hermetic simulation)"
	print "COPY " staging "/cachi2 /cachi2"
	next
}
{ print }
' "$base" | awk -v staging="$STAGING_REL" '
$0 ~ /^COPY LICENSE \/licenses\/LICENSE$/ {
	print
	print ""
	print "# Hermeto project_files (file:// wheel paths) overlay"
	print "COPY " staging "/patched-requirements/ ./"
	next
}
{ print }
'
