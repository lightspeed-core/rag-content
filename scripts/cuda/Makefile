# CUDA lab — local checks and AWS integration helpers for provision_cuda_lab.sh
# and related scripts. Run from anywhere: make -C scripts/cuda help
#
# Integration targets need: AWS CLI, credentials, EC2 key pair LAB_USER-keys in REGION,
# and private key file (see provision_cuda_lab.sh). Costs apply (GPU instances).

SHELL := /bin/bash

CUDA_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROVISION := $(CUDA_DIR)provision_cuda_lab.sh
INSTALL := $(CUDA_DIR)install_cuda_rhel9_ec2.sh
CFN := $(CUDA_DIR)cuda_cloud_formation.yaml

# --- integration (override on command line) ---
LAB_USER ?=
REGION ?= us-east-1
STACK_NAME ?=
ALLOWED_CIDR ?=
KEY_FILE ?=

.PHONY: help
help:
	@echo "Local (no GPU cost):"
	@echo "  make check-syntax          - bash -n on provision + install scripts"
	@echo "  make check-shellcheck      - shellcheck both scripts if shellcheck is installed"
	@echo "  make validate-cfn          - aws cloudformation validate-template"
	@echo "  make test-opt-guards       - invalid CLI args must exit non-zero"
	@echo "  make test-local            - check-syntax + validate-cfn + check-shellcheck"
	@echo ""
	@echo "AWS integration (costs / side effects):"
	@echo "  make stack-only LAB_USER=<u> [REGION=...] [STACK_NAME=...] [ALLOWED_CIDR=...] [KEY_FILE=...]"
	@echo "  make lab-full   LAB_USER=<u>  ...  # full provision + CUDA install + wait"
	@echo "  make reuse-install LAB_USER=<u> [REGION=...] [STACK_NAME=...]  # --reuse-stack"
	@echo "  make delete-stack STACK_NAME=<name> [REGION=...]"
	@echo ""
	@echo "Examples:"
	@echo "  make -C scripts/cuda test-local"
	@echo "  make -C scripts/cuda stack-only LAB_USER=\$$USER REGION=us-east-1"

.PHONY: check-syntax
check-syntax: check-syntax-provision check-syntax-install

.PHONY: check-syntax-provision
check-syntax-provision:
	bash -n "$(PROVISION)"

.PHONY: check-syntax-install
check-syntax-install:
	bash -n "$(INSTALL)"

.PHONY: check-shellcheck
check-shellcheck: shellcheck-provision shellcheck-install

.PHONY: shellcheck-provision
shellcheck-provision:
	@if command -v shellcheck >/dev/null 2>&1; then \
		shellcheck "$(PROVISION)"; \
	else \
		echo "shellcheck not installed; skipping $(PROVISION)"; \
	fi

.PHONY: shellcheck-install
shellcheck-install:
	@if command -v shellcheck >/dev/null 2>&1; then \
		shellcheck "$(INSTALL)"; \
	else \
		echo "shellcheck not installed; skipping $(INSTALL)"; \
	fi

.PHONY: validate-cfn
validate-cfn:
	aws cloudformation validate-template --template-body "file://$(CFN)" >/dev/null
	@echo "validate-cfn: OK"

# Invalid invocations must fail (opt_arg_or_usage / usage).
.PHONY: test-opt-guards
test-opt-guards: check-syntax-provision
	@echo "test-opt-guards: expecting failures..."
	@! "$(PROVISION)" >/dev/null 2>&1
	@! "$(PROVISION)" myuser --region >/dev/null 2>&1
	@! "$(PROVISION)" myuser --region --reuse-stack >/dev/null 2>&1
	@echo "test-opt-guards: OK"

.PHONY: test-local
test-local: check-syntax validate-cfn check-shellcheck test-opt-guards
	@echo "test-local: all non-AWS checks passed."

# --- integration ---

.PHONY: require-lab-user
require-lab-user:
	@test -n "$(LAB_USER)" || { echo "error: set LAB_USER=<username> (EC2 key pair prefix: \$${LAB_USER}-keys)" >&2; exit 1; }

_stack_opts := $(PROVISION) "$(LAB_USER)" --region "$(REGION)" \
	$(if $(STACK_NAME),--stack-name $(STACK_NAME),) \
	$(if $(ALLOWED_CIDR),--allowed-cidr $(ALLOWED_CIDR),) \
	$(if $(KEY_FILE),--key-file $(KEY_FILE),)

.PHONY: stack-only
stack-only: require-lab-user
	$(_stack_opts) --stack-only

.PHONY: lab-full
lab-full: require-lab-user
	$(_stack_opts)

.PHONY: reuse-install
reuse-install: require-lab-user
	$(_stack_opts) --reuse-stack

.PHONY: delete-stack
delete-stack:
	@test -n "$(STACK_NAME)" || { echo "error: set STACK_NAME=<stack> (e.g. \$${LAB_USER}-cuda-stack)" >&2; exit 1; }
	aws cloudformation delete-stack --stack-name "$(STACK_NAME)" --region "$(REGION)"
	@echo "delete-stack: submitted. Wait with: aws cloudformation wait stack-delete-complete --stack-name $(STACK_NAME) --region $(REGION)"
