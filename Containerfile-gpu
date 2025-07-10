ARG FLAVOR=cpu

FROM registry.access.redhat.com/ubi9/python-311 as cpu-base
ARG FLAVOR

FROM nvcr.io/nvidia/cuda:12.6.2-devel-ubi9 as gpu-base
ARG FLAVOR
RUN dnf install -y python3.11 python3.11-pip libcudnn8 libnccl git
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH

FROM ${FLAVOR}-base as lightspeed-core-rag-builder
ARG FLAVOR

USER 0
RUN dnf install -y rubygems && \
    dnf clean all && \
    gem install asciidoctor

WORKDIR /rag-content
ENV EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

COPY . /rag-content
RUN make install-global

# Test torch
RUN if [[ $(echo $LD_LIBRARY_PATH) == *"/usr/local/cuda-12.6/compat"* ]]; then \
        python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available());"; \
    fi

# # Download embeddings model
RUN python ./scripts/download_embeddings_model.py \
        -l ./embeddings_model \
        -r ${EMBEDDING_MODEL}

LABEL description="Contains embedding model and dependencies needed to generate a vector database"
