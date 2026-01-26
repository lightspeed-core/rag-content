# Image with CPU only backend. Smaller images.
FROM registry.access.redhat.com/ubi9/ubi-minimal

# Install Python and Ruby
RUN microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.12 python3.12-devel python3.12-pip git \
    rubygems rubygem-bundler && \
    microdnf clean all

# Install uv package manager
RUN pip3.12 install uv>=0.7.20

WORKDIR /rag-content

COPY Makefile pyproject.toml uv.lock README.md Gemfile Gemfile.lock requirements.hashes.wheel.txt requirements.hashes.source.txt requirements-build.txt ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts

# Install Ruby Gems
RUN BUNDLE_PATH__SYSTEM=true bundle install

# Configure UV environment variables for optimal performance
# Pytorch backend - cpu. `uv` contains convenient way to specify the backend.
# MATURIN_NO_INSTALL_RUST=1 : Disable installation of Rust dependencies by Maturin.
ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    MATURIN_NO_INSTALL_RUST=1

# If Cachi2 is present, use pip to install dependencies, otherwise use uv.
RUN if [ -f /cachi2/cachi2.env ]; then \
    . /cachi2/cachi2.env && \
    uv venv --seed --no-index --find-links ${PIP_FIND_LINKS} && \
    . .venv/bin/activate && \
    pip install --no-cache-dir --ignore-installed --no-index --find-links ${PIP_FIND_LINKS} --no-deps -r requirements.hashes.wheel.txt -r requirements.hashes.source.txt && \
    pip check; \
    else \
    uv sync --locked --no-dev; \
    fi

# Add executables from .venv to system PATH
ENV PATH="/rag-content/.venv/bin:$PATH"

# Download embeddings model
ENV EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
RUN python ./scripts/download_embeddings_model.py \
    -l ./embeddings_model \
    -r ${EMBEDDING_MODEL}

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