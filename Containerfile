ARG BUILDER_BASE_IMAGE=registry.access.redhat.com/ubi9/ubi-minimal
ARG RUNTIME_BASE_IMAGE=registry.access.redhat.com/ubi9/ubi-minimal

# Stage 1: Builder — install build tools, compile Python deps from sdist, then discard.
FROM ${BUILDER_BASE_IMAGE} AS builder
ARG DNF_COMMAND=microdnf
ARG TARGETARCH
USER root

# Install Python and git (always needed).
RUN ${DNF_COMMAND} install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.12 python3.12-devel python3.12-pip git && \
    ${DNF_COMMAND} clean all

# Hermetic: install build toolchain and dev libs for compiling sdists.
RUN if [ -f /cachi2/cachi2.env ]; then \
    ${DNF_COMMAND} install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    gcc cmake git libpq-devel swig autoconf automake libtool libxml2-devel libxslt-devel && \
    ${DNF_COMMAND} clean all; \
    fi

# Install uv package manager
RUN pip3.12 install uv>=0.7.20

WORKDIR /rag-content

COPY Makefile pyproject.toml uv.lock README.md Gemfile Gemfile.lock .konflux/requirements.hashes.wheel.txt .konflux/requirements.hashes.wheel.pypi.txt .konflux/requirements.hashes.source.txt .konflux/requirements-build.txt ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY embeddings_model ./embeddings_model
COPY LICENSE /licenses/LICENSE

# Install Ruby Gems
RUN BUNDLE_PATH__SYSTEM=true bundle install

# Configure UV environment variables for optimal performance
# Pytorch backend - cpu. `uv` contains convenient way to specify the backend.
# MATURIN_NO_INSTALL_RUST=1 : Disable installation of Rust dependencies by Maturin.
ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    MATURIN_NO_INSTALL_RUST=1

WORKDIR /rag-content

COPY pyproject.toml uv.lock README.md .konflux/requirements.hashes.wheel.txt .konflux/requirements.hashes.wheel.cpu.x86_64.txt .konflux/requirements.hashes.wheel.cpu.aarch64.txt .konflux/requirements.hashes.wheel.pypi.txt .konflux/requirements.hashes.source.txt .konflux/requirements-build.txt ./
COPY src ./src

RUN if [ -f /cachi2/cachi2.env ]; then \
    . /cachi2/cachi2.env && \
    uv venv --seed --no-index --find-links ${PIP_FIND_LINKS} && \
    . .venv/bin/activate && \
    case "${TARGETARCH:-amd64}" in amd64) CPU_WHEEL_ARCH=x86_64 ;; arm64) CPU_WHEEL_ARCH=aarch64 ;; *) CPU_WHEEL_ARCH=x86_64 ;; esac && \
    pip install --no-cache-dir --ignore-installed --no-index --find-links ${PIP_FIND_LINKS} --no-deps \
      -r requirements.hashes.wheel.txt \
      -r requirements.hashes.wheel.pypi.txt \
      -r requirements.hashes.source.txt && \
    pip install --no-cache-dir --no-deps . && \
    pip check; \
    else \
    uv sync --locked --no-dev; \
    fi

# Stage 2: Runtime — clean image with only runtime dependencies.
FROM ${RUNTIME_BASE_IMAGE}
ARG DNF_COMMAND=microdnf
USER root

RUN ${DNF_COMMAND} install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.12 python3.12-pip git \
    libpq libxml2 libxslt libjpeg-turbo libtiff freetype libwebp \
    rubygems rubygem-bundler \
    skopeo && \
    ${DNF_COMMAND} clean all

WORKDIR /rag-content

# Copy the built venv from the builder stage.
COPY --from=builder --chown=1000:1000 /rag-content/.venv .venv

COPY Makefile pyproject.toml uv.lock README.md Gemfile Gemfile.lock ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY embeddings_model ./embeddings_model
COPY LICENSE /licenses/LICENSE

# Install Ruby Gems
RUN BUNDLE_PATH__SYSTEM=true bundle install

ENV PATH="/rag-content/.venv/bin:$PATH"

# Download embeddings model
# In hermetic build, the model is already downloaded and mounted to the container.
ENV EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
RUN if [ -f /cachi2/cachi2.env ]; then \
    mkdir -p embeddings_model && \
    cp /cachi2/output/deps/generic/model.safetensors embeddings_model/model.safetensors; \
    else \
    python ./scripts/download_embeddings_model.py \
    -l ./embeddings_model \
    -r ${EMBEDDING_MODEL}; \
    fi

# Create non-root user and set ownership of app directory
RUN groupadd -r rag -g 1000 && \
    useradd -r -u 1000 -g rag -d /rag-content -s /sbin/nologin rag && \
    chown -R rag:rag /rag-content

# Run as non-root user
USER 1000

# Reset the entrypoint.
ENTRYPOINT []

LABEL vendor="Red Hat, Inc." \
    name="lightspeed-core/rag-tool-rhel9" \
    com.redhat.component="lightspeed-core/rag-tool" \
    cpe="cpe:/a:redhat:lightspeed_core:0.4::el9" \
    io.k8s.display-name="Lightspeed RAG Tool" \
    summary="RAG tool containing embedding model and dependencies needed to generate a vector database." \
    description="RAG Tool provides a shared codebase for generating vector databases. It serves as the core framework for Lightspeed-related projects (e.g., OpenShift Lightspeed, OpenStack Lightspeed, etc.) to generate their own vector databases that can be used for RAG." \
    io.k8s.description="RAG Tool provides a shared codebase for generating vector databases. It serves as the core framework for Lightspeed-related projects (e.g., OpenShift Lightspeed, OpenStack Lightspeed, etc.) to generate their own vector databases that can be used for RAG." \
    io.openshift.tags="lightspeed-core,lightspeed-rag-tool,lightspeed"
