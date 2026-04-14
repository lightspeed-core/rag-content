NUM_WORKERS ?= $$(( $(shell nproc --all) / 2))
ARTIFACT_DIR := $(if $(ARTIFACT_DIR),$(ARTIFACT_DIR),tests/test_results)

# Define arguments for pgvector support
POSTGRES_USER ?= postgres
POSTGRES_PASSWORD ?= somesecret
POSTGRES_HOST ?= localhost
POSTGRES_PORT ?= 15432
POSTGRES_DATABASE ?= postgres

# Container image build / output arguments
TOOL_IMAGE ?= localhost/rag-content:latest
DOCS_DIR ?= docs
OUTPUT_DIR ?= output
INDEX_NAME ?= my-index
OUTPUT_IMAGE_NAME ?= rag-content-output
OUTPUT_IMAGE_TAG ?= latest
OUTPUT_IMAGE_FILE ?= $(OUTPUT_DIR)/$(OUTPUT_IMAGE_NAME)-$(OUTPUT_IMAGE_TAG).tar


.PHONY: unit-test
test-unit: ## Run the unit tests
	@echo "Running unit tests..."
	@echo "Reports will be written to ${ARTIFACT_DIR}"
	COVERAGE_FILE="${ARTIFACT_DIR}/.coverage.unit" uv run pytest tests --cov=src/lightspeed_rag_content --cov-report term-missing --cov-report "json:${ARTIFACT_DIR}/coverage_unit.json" --junit-xml="${ARTIFACT_DIR}/junit_unit.xml" --cov-fail-under=60

.PHONY: install-tools
install-tools: ## Install required utilities/tools
	@command -v uv > /dev/null || { echo >&2 "uv is not installed. Installing..."; pip3.12 install --upgrade pip uv; }

.PHONY: uv-lock-check
uv-lock-check: ## Check that the uv.lock file is in a good shape
	uv lock --check

.PHONY: install-global
install-global: install-tools ## Install ligthspeed-rag-content into file system.
	uv pip install --python 3.12 --system .

.PHONY: install-hooks
install-hooks: install-deps-test ## Install commit hooks
	uv pip install pre-commit

.PHONY: install-deps
install-deps: install-tools uv-lock-check ## Install all required dependencies, according to uv.lock
	uv sync

.PHONY: install-deps-test
install-deps-test: install-tools uv-lock-check ## Install all required dev dependencies, according to uv.lock
	uv sync --dev

.PHONY: update-deps
update-deps: ## Check pyproject.toml for changes, update the lock file if needed, then sync.
	uv lock --upgrade
	uv sync
	uv sync --dev

.PHONY: check-types
check-types: ## Check types in the code.
	@echo "Running $@ target ..."
	uv run mypy --namespace-packages --explicit-package-bases --strict --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs src scripts

.PHONY: check-format
check-format: ## Check that the code is properly formatted using Black and Ruff formatter.
	@echo "Running $@ target ..."
	uv run black --check scripts src tests
	uv run ruff check scripts src

.PHONY: check-coverage
check-coverage: ## Check the coverage of unit tests.
	@echo "Running $@ target ..."
	uv run coverage run --source=src/lightspeed_rag_content -m pytest tests --verbose && uv run coverage report -m --fail-under 90

.PHONY: check-code-metrics
check-code-metrics: ## Check the code using Radon.
	@echo "Running $@ target ..."
	@RADON_OUT=$$(uv run radon cc -a src/) && \
	echo "$$RADON_OUT" && \
	OUTPUT=$$(echo "$$RADON_OUT" | tail -1) && \
	GRADE=$$(echo "$$OUTPUT" | awk '/Average complexity:/ { print $$3; exit }') && \
	if [ "$$GRADE" = "A" ]; then exit 0; else exit 1; fi

.PHONY: format
format: ## Format the code into unified format
	uv run black scripts src tests
	uv run ruff check scripts src --fix
	uv run pre-commit run

black:
	uv tool run black --check .

.PHONY: pylint
pylint: ## Run pylint on application source.
	@echo "Running $@ target ..."
	uv run pylint src

ruff:
	uv run ruff check src

.PHONY: verify
verify: check-types check-format pylint check-code-metrics check-coverage ## Verify the code using various linters

.PHONY: start-postgres
start-postgres: ## Start postgresql from the pgvector container image
	mkdir -pv ./postgresql/data ./output
	podman run -d --name pgvector --rm -e POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
	 -p $(POSTGRES_PORT):5432 \
	 -v $(PWD)/postgresql/data:/var/lib/postgresql/data:Z pgvector/pgvector:pg16

.PHONY: start-postgres-debug
start-postgres-debug: ## Start postgresql from the pgvector container image with debugging enabled
	mkdir -pv ./postgresql/data ./output
	podman run --name pgvector --rm -e POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
	 -p $(POSTGRES_PORT):5432 \
	 -v ./postgresql/data:/var/lib/postgresql/data:Z pgvector/pgvector:pg16 \
	 postgres -c log_statement=all -c log_destination=stderr

.PHONY: build-tool-image
build-tool-image: ## Build the rag-content tool image (uses the existing Containerfile)
	podman build -t $(TOOL_IMAGE) .

.PHONY: generate-output-image
# skopeo (needed to fetch the base image) is bundled in the tool container but
# not required on the host, so the script is run inside the container.
generate-output-image: build-tool-image ## Run the tool container and produce a vector-DB image archive
	mkdir -p "$(OUTPUT_DIR)"
	outfile="$$(basename "$(OUTPUT_IMAGE_FILE)")"; \
	podman run --rm \
	  -v "$(abspath $(DOCS_DIR)):/input:ro,Z" \
	  -v "$(abspath $(OUTPUT_DIR)):/output:Z" \
	  "$(TOOL_IMAGE)" \
	  python /rag-content/scripts/generate_embeddings.py \
	    -f /input \
	    -o /output/vector_db \
	    -i "$(INDEX_NAME)" \
	    --output-image "/output/$$outfile" \
	    --image-name "$(OUTPUT_IMAGE_NAME)" \
	    --image-tag "$(OUTPUT_IMAGE_TAG)"
	@echo ""
	@echo "Image archive written to $(OUTPUT_IMAGE_FILE)"
	@echo "Load with:  podman load < $(OUTPUT_IMAGE_FILE)"
	@echo "Then push:  podman push $(OUTPUT_IMAGE_NAME):$(OUTPUT_IMAGE_TAG)"

konflux-requirements:	## generate hermetic requirements.*.txt file and gemfile.lock for konflux build
	./scripts/konflux_requirements.sh
	bundle _2.2.33_ lock --add-platform aarch64-linux

HERMETO_MODE ?= pip-cpu
hermeto-fetch-deps:	## run Hermeto prefetch (HERMETO_MODE=pip-cpu|pip-cuda|full-cpu|full-cuda); podman/docker + network
	@./scripts/run_hermeto_fetch_deps.sh "$(HERMETO_MODE)"

# Pip-slice checks against committed lockfiles (mirrors Konflux prefetch-dependencies for Python).
# CPU success does not imply CUDA: different requirements.*.cuda.txt, overrides.cuda.txt, and wheel/source splits
# (e.g. Hermeto runs cargo vendor --locked on some sdists only seen in one pipeline).
.PHONY: hermeto-verify-pip-cpu hermeto-verify-pip-cuda hermeto-verify-pip
hermeto-verify-pip-cpu: ## Hermeto strict fetch-deps pip-cpu → .hermeto-output-verify-cpu
	HERMETO_OUT="$(CURDIR)/.hermeto-output-verify-cpu" ./scripts/run_hermeto_fetch_deps.sh pip-cpu
hermeto-verify-pip-cuda: ## Hermeto strict fetch-deps pip-cuda → .hermeto-output-verify-cuda
	HERMETO_OUT="$(CURDIR)/.hermeto-output-verify-cuda" ./scripts/run_hermeto_fetch_deps.sh pip-cuda
hermeto-verify-pip: hermeto-verify-pip-cpu hermeto-verify-pip-cuda ## both pip slices (run before pushing Konflux prefetch changes)

konflux-rpm-lock:	## generate rpm.lock.yaml file for konflux build
	./scripts/generate-rpm-lock.sh

ruby-bundler: # Install bundler 2.2.33, this is the version used by the container image.
	gem install bundler -v 2.2.33

.PHONY: help
help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ''
