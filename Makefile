# Default to CPU if not specified
FLAVOR ?= cpu
NUM_WORKERS ?= $$(( $(shell nproc --all) / 2))

# Define behavior based on the flavor
ifeq ($(FLAVOR),cpu)
TORCH_GROUP := cpu
else ifeq ($(FLAVOR),gpu)
TORCH_GROUP := gpu
else
$(error Unsupported FLAVOR $(FLAVOR), must be 'cpu' or 'gpu')
endif

# Define arguments for pgvector support
POSTGRES_USER ?= postgres
POSTGRES_PASSWORD ?= somesecret
POSTGRES_HOST ?= localhost
POSTGRES_PORT ?= 15432
POSTGRES_DATABASE ?= postgres

install-tools: ## Install required utilities/tools
	@command -v pdm > /dev/null || { echo >&2 "pdm is not installed. Installing..."; pip3.11 install --upgrade pip pdm; }

pdm-lock-check: ## Check that the pdm.lock file is in a good shape
	pdm lock --check --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

install-global: install-tools pdm-lock-check ## Install ligthspeed-rag-content to global Python directories
	pdm install --global --project . --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

install-hooks: install-deps-test ## Install commit hooks
	pdm run pre-commit install

install-deps: install-tools pdm-lock-check ## Install all required dependencies, according to pdm.lock
	pdm sync --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

install-deps-test: install-tools pdm-lock-check ## Install all required dev dependencies, according to pdm.lock
	pdm sync --dev --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

update-deps: ## Check pyproject.toml for changes, update the lock file if needed, then sync.
	pdm install --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)
	pdm install --dev --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

check-types: ## Checks type hints in sources
	pdm run mypy --explicit-package-bases --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs scripts

format: ## Format the code into unified format
	pdm run black scripts src
	pdm run ruff check scripts src --fix --per-file-ignores=scripts/*:S101
	pdm run pre-commit run

verify: check-types ## Verify the code using various linters
	pdm run black --check scripts src
	pdm run ruff check scripts src --per-file-ignores=scripts/*:S101

update-docs: ## Update the plaintext OCP docs in ocp-product-docs-plaintext/
	@set -e && for OCP_VERSION in $$(ls -1 ocp-product-docs-plaintext); do \
		examples/get_ocp_plaintext_docs.sh $$OCP_VERSION; \
	done
	examples/get_runbooks.sh

build-image-ocp-example: build-base-image ## Build a rag-content container image
	podman build -t ocp-rag-content -f examples/Containerfile.ocp_lightspeed --build-arg FLAVOR=$(TORCH_GROUP) --build-arg NUM_WORKERS=$(NUM_WORKERS) .

build-image-ocp-example-test: build-image-ocp-example ## Build test image for the OCP example
	podman build -t test-rag-content -f examples/Containerfile.minimal_test .

run-ocp-example-test: build-image-ocp-example-test ## Execute test image for the OCP example
	podman run test-rag-content

build-base-image: ## Build base container image
	podman build -t $(TORCH_GROUP)-road-core-base -f Containerfile.base --build-arg FLAVOR=$(TORCH_GROUP)

start-postgres: ## Start postgresql from the pgvector container image
	mkdir -pv ./postgresql/data ./output
	podman run -d --name pgvector --rm -e POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
	 -p $(POSTGRES_PORT):5432 \
	 -v $(PWD)/postgresql/data:/var/lib/postgresql/data:Z pgvector/pgvector:pg16

start-postgres-debug: ## Start postgresql from the pgvector container image with debugging enabled
	mkdir -pv ./postgresql/data ./output
	podman run --name pgvector --rm -e POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
	 -p $(POSTGRES_PORT):5432 \
	 -v ./postgresql/data:/var/lib/postgresql/data:Z pgvector/pgvector:pg16 \
	 postgres -c log_statement=all -c log_destination=stderr

generate-embeddings-postgres: ## Generate embeddings for postgres vector store
	POSTGRES_USER=$(POSTGRES_USER) \
	POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
	POSTGRES_HOST=$(POSTGRES_HOST) \
	POSTGRES_PORT=$(POSTGRES_PORT) \
	POSTGRES_DATABASE=$(POSTGRES_DATABASE) \
	pdm run python examples/generate_embeddings_openshift.py \
	 -o ./output \
	 -f ocp-product-docs-plaintext/4.15/ \
	 -r runbooks/ \
	 -md embeddings_model/ \
	 -mn sentence-transformers/all-mpnet-base-v2 \
	 -v 4.15 \
	 -i  ocp-product-docs-4_15 \
	 --vector-store-type postgres

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ''
