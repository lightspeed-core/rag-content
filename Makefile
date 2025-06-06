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

.PHONY: install-tools
install-tools: ## Install required utilities/tools
	@command -v pdm > /dev/null || { echo >&2 "pdm is not installed. Installing..."; pip3.11 install --upgrade pip pdm; }

.PHONY: pdm-lock-check
pdm-lock-check: ## Check that the pdm.lock file is in a good shape
	pdm lock --check --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

.PHONY: install-glob
install-global: install-tools pdm-lock-check ## Install ligthspeed-rag-content to global Python directories
	pdm install --global --project . --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

.PHONY: install-hooks
install-hooks: install-deps-test ## Install commit hooks
	pdm run pre-commit install

.PHONY: install-deps
install-deps: install-tools pdm-lock-check ## Install all required dependencies, according to pdm.lock
	pdm sync --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

.PHONY: install-deps-test
install-deps-test: install-tools pdm-lock-check ## Install all required dev dependencies, according to pdm.lock
	pdm sync --dev --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

.PHONY: update-deps
update-deps: ## Check pyproject.toml for changes, update the lock file if needed, then sync.
	pdm install --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)
	pdm install --dev --group $(TORCH_GROUP) --lockfile pdm.lock.$(TORCH_GROUP)

.PHONY: check-types
check-types: ## Check types in the code.
	@echo "Running $@ target ..."
	pdm run mypy --namespace-packages --explicit-package-bases --strict --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs src scripts

.PHONY: check-format
check-format: ## Check that the code is properly formatted using Black and Ruff formatter.
	@echo "Running $@ target ..."
	pdm run black --check scripts src
	pdm run ruff check scripts src --per-file-ignores=scripts/*:S101

.PHONY: check-coverage
check-coverage: ## Check the coverage of unit tests.
	@echo "Running $@ target ..."
	pdm run test

.PHONY: check-code-metrics
check-code-metrics: ## Check the code using Radon.
	@echo "Running $@ target ..."
	@OUTPUT=$$(pdm run radon cc -a A src/ | tee /dev/tty | tail -1) && \
	GRADE=$$(echo $$OUTPUT | grep -oP " [A-F] " | tr -d '[:space:]') && \
	if [ "$$GRADE" = "A" ]; then exit 0; else exit 1; fi

.PHONY: format
format: ## Format the code into unified format
	pdm run black scripts src
	pdm run ruff check scripts src --fix --per-file-ignores=scripts/*:S101
	pdm run pre-commit run

black:
	pdm run black --check .

pylint:
	pdm run pylint src

ruff:
	pdm run ruff check src --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101

.PHONY: verify
verify: check-types check-format check-code-metrics check-coverage ## Verify the code using various linters

.PHONY: build-base-image
build-base-image: ## Build base container image
	podman build -t $(TORCH_GROUP)-lightspeed-core-base -f Containerfile --build-arg FLAVOR=$(TORCH_GROUP)

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

.PHONY: help
help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ''
