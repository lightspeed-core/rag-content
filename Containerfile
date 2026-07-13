ARG BUILDER_BASE_IMAGE=registry.access.redhat.com/ubi9/ubi-minimal
ARG RUNTIME_BASE_IMAGE=registry.access.redhat.com/ubi9/ubi-minimal

# Stage 1: Builder — install build tools, compile Python deps from sdist, then discard.
FROM ${BUILDER_BASE_IMAGE} AS builder
ARG BUILDER_DNF_COMMAND=microdnf
ARG TARGETARCH
USER root

# Install Python and build tools.
RUN ${BUILDER_DNF_COMMAND} install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.12 python3.12-devel python3.12-pip && \
    ${BUILDER_DNF_COMMAND} clean all


# Install uv package manager
RUN pip3.12 install "uv>=0.7.20"

WORKDIR /rag-content

# Configure UV environment variables for optimal performance
# Pytorch backend - cpu. `uv` contains convenient way to specify the backend.
# MATURIN_NO_INSTALL_RUST=1 : Disable installation of Rust dependencies by Maturin.
ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    MATURIN_NO_INSTALL_RUST=1

COPY pyproject.toml uv.lock README.md .konflux/requirements.hashes.wheel.txt .konflux/requirements.hashes.wheel.pypi.txt .konflux/requirements.hashes.source.txt .konflux/requirements-build.txt ./
COPY src ./src
COPY LICENSE /licenses/LICENSE

RUN if [ -f /cachi2/cachi2.env ]; then \
    . /cachi2/cachi2.env && \
    python3.12 -c "import os,re;d=os.environ['PIP_FIND_LINKS'];fs=os.listdir(d);rp={re.split(r'-\d+-(?:cp|py|pp)',f)[0] for f in fs if f.endswith('.whl') and re.search(r'-\d+-(?:cp|py|pp)',f)};[os.remove(os.path.join(d,f)) for f in fs if f.endswith('.whl') and not re.search(r'-\d+-(?:cp|py|pp)',f) and f.rsplit('-',3)[0] in rp]" && \
    uv venv --python python3.12 && \
    . .venv/bin/activate && \
    sed -i '/^--index-url/d' requirements.hashes.wheel.txt requirements.hashes.wheel.pypi.txt requirements.hashes.source.txt && \
    uv pip install --no-cache --reinstall --no-index --find-links ${PIP_FIND_LINKS} --no-deps \
      -r requirements.hashes.wheel.txt \
      -r requirements.hashes.wheel.pypi.txt \
      -r requirements.hashes.source.txt && \
    uv pip check; \
    else \
    uv sync --locked --no-dev; \
    fi

# Stage 2: Runtime — clean image with only runtime dependencies.
FROM ${RUNTIME_BASE_IMAGE}
ARG RUNTIME_DNF_COMMAND=microdnf
USER root

RUN ${RUNTIME_DNF_COMMAND} -y module enable ruby:3.3 && \
    ${RUNTIME_DNF_COMMAND} install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.12 \
    libpq libxml2 libxslt libjpeg-turbo libtiff freetype libwebp \
    rubygems rubygem-bundler \
    skopeo && \
    ${RUNTIME_DNF_COMMAND} update -y --nodocs && \
    ${RUNTIME_DNF_COMMAND} clean all

WORKDIR /rag-content

# Copy the built venv from the builder stage.
COPY --from=builder --chown=1000:1000 /rag-content/.venv .venv

# Install asciidoctor gem (for .adoc preprocessing)
COPY Gemfile Gemfile.lock ./
RUN if [ -f /cachi2/cachi2.env ]; then . /cachi2/cachi2.env; fi && \
    BUNDLE_PATH__SYSTEM=true bundle install

COPY src ./src
COPY scripts/generate_embeddings.py scripts/download_embeddings_model.py ./scripts/
COPY embeddings_model ./embeddings_model
COPY LICENSE /licenses/LICENSE

ENV PATH="/rag-content/.venv/bin:$PATH" \
    PYTHONPATH="/rag-content/src"

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
    name="lightspeed-core/rag-tool-cpu-rhel9" \
    com.redhat.component="lightspeed-core/rag-tool-cpu-rhel9" \
    cpe="cpe:/a:redhat:lightspeed_core:0.7::el9" \
    io.k8s.display-name="Lightspeed RAG Tool (CPU)" \
    summary="RAG tool (CPU) containing embedding model and dependencies needed to generate a vector database." \
    description="RAG Tool (CPU) provides a shared codebase for generating vector databases. It serves as the core framework for Lightspeed-related projects (e.g., OpenShift Lightspeed, OpenStack Lightspeed, etc.) to generate their own vector databases that can be used for RAG." \
    io.k8s.description="RAG Tool (CPU) provides a shared codebase for generating vector databases. It serves as the core framework for Lightspeed-related projects (e.g., OpenShift Lightspeed, OpenStack Lightspeed, etc.) to generate their own vector databases that can be used for RAG." \
    io.openshift.tags="lightspeed-core,lightspeed-rag-tool-cpu,lightspeed"
