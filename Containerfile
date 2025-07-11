# Image with CPU only backend. Smaller images.
FROM registry.access.redhat.com/ubi9/ubi-minimal

# Install Python
RUN microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.11 python3.11-devel python3.11-pip && \
    microdnf clean all
# Install asciidoctor
RUN microdnf install -y rubygems && \
    microdnf clean all && \
    gem install asciidoctor
# Install uv package manager
RUN pip3.11 install uv==0.7.20

WORKDIR /rag-content

COPY Makefile pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts

# Configure UV environment variables for optimal performance
# Pytorch backend - cpu. `uv` contains convenient way to specify the backend.
ENV UV_COMPILE_BYTECODE=0 \
    UV_PYTHON_DOWNLOADS=0

# Install Python dependencies
RUN uv sync --locked --no-install-project

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
RUN uv sync --locked

# Add executables from .venv to system PATH
ENV PATH="/rag-content/.venv/bin:$PATH"

# Download embeddings model
ENV EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
RUN python ./scripts/download_embeddings_model.py \
       -l ./embeddings_model \
       -r ${EMBEDDING_MODEL}

# Reset the entrypoint.
ENTRYPOINT []

LABEL description="Contains embedding model and dependencies needed to generate a vector database"
